# RGB-D 深度融合代码修改计划（简化版）

## 1. 修改范围

本计划只包含准备直接落地的代码修改和对应测试，不包含：

- 数据统计脚本；
- 注意力消融实验；
- 长训练安排；
- 局部窗口或对角残差；
- 独立位置编码；
- 深度 MLP、卷积 stem 或 self-attention；
- 深度梯度和法向输入。

这些内容在 review 中都属于需要实验验证的候选方案，不是当前必须修改的代码。

本轮最主要的模型修改是 `MetricDepthEncoder` 的有效性通道。

## 2. 修改一：深度编码器增加 validity 通道

### 2.1 当前问题

当前 `MetricDepthEncoder` 把深度归一化到 `[-1, 1]`，再把无效像素填成 `0.0`。

在当前 log 深度范围下，归一化值 `0.0` 同时可能表示：

- 无效或缺失深度；
- 真实的约 `0.5m` 深度。

当一个 patch 的无效比例低于 `invalid_threshold=0.5` 时，该 patch 仍然进入 K/V，卷积无法区分其中的无效像素和真实中距离像素。

### 2.2 修改文件

- `turbovla/models/depth_encoder.py`
- `tests/test_depth_modules.py`

### 2.3 修改方式

保留当前深度范围和 mask 行为：

```python
valid = torch.isfinite(depth_m)
valid = valid & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
```

也就是说，本轮不改变“大于 5 米视为无效”的默认语义。

将卷积输入从一个通道改为两个通道：

```text
channel 0 = normalized_depth
channel 1 = valid.float()
```

对应修改：

```python
self.patch_embed = nn.Conv2d(
    in_channels=2,
    out_channels=config.hidden_dim,
    kernel_size=config.patch_size,
    stride=config.patch_size,
)
```

在 `forward()` 中拼接：

```python
encoder_input = torch.cat(
    [normalized, valid_pixels.to(dtype=normalized.dtype)],
    dim=2,
)
```

然后把 `encoder_input` 送入 `patch_embed`。

token 级 `invalid_mask` 仍然由当前 `valid_pixels` 和 `invalid_threshold` 计算，不改变 cross-attention mask 规则。

### 2.4 初始化和 checkpoint 约束

深度分支目前是新增随机初始化模块，RGB-only checkpoint 不包含 `depth_encoder.patch_embed.weight`，因此把输入从一通道改成两通道不会影响旧 RGB checkpoint 加载。

本计划假设当前没有必须继续使用的已训练一通道 RGB-D checkpoint。若存在此类 checkpoint，需要额外增加一通道到两通道的权重迁移：

```text
new_weight[:, 0] = old_weight[:, 0]
new_weight[:, 1] = 0
```

无论采用哪种初始化，`gate_init=0` 时完整模型输出仍必须与 RGB-only 模型逐元素相等。

### 2.5 测试

新增或修改测试：

1. `patch_embed.in_channels == 2`；
2. 完全有效深度能够正常编码；
3. 全零深度生成全无效 patch mask；
4. 部分无效 patch 的 validity 通道内容正确；
5. 真实归一化中点和无效像素在第二通道上可以区分；
6. 输出 token 和 invalid mask 形状保持不变；
7. gate 为零时完整模型仍严格等于 RGB-only 输出。

## 3. 修改二：增加必要的防误配校验

这些修改不改变网络结构，只让错误数据提前失败。

### 3.1 深度单位校验

涉及文件：

- `third_party/starvla_runtime/starVLA/dataloader/__init__.py`
- `third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/datasets.py`

在创建 `DataLoader` 前检查所有子数据集：

- 深度元数据单位必须存在；
- 三个深度相机的单位必须一致；
- 数据单位必须等于 `cfg.framework.depth.input_unit`。

单位不一致时直接抛出 `ValueError`。

### 3.2 RGB/depth 相机顺序校验

把下面两组 key 去掉模态前缀后逐项比较：

```text
video.cam_high       <-> depth.cam_high
video.cam_left_wrist <-> depth.cam_left_wrist
video.cam_right_wrist <-> depth.cam_right_wrist
```

校验数量、顺序和规范化后的相机 ID。任何不一致都直接抛错。

### 3.3 图像处理器尺寸校验

涉及文件：

- `third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py`

修改内容：

1. 若处理器支持 `do_center_crop`，显式设为 `False`；
2. 保持 resize 为配置的 `image_size`；
3. 在 `_model_inputs()` 中断言：

```python
pixel_values.shape[-2:] == (self.image_size, self.image_size)
```

## 4. 修改三：增加最小 gate 监控

涉及文件：

- `third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py`

只增加一个低成本指标：

```text
depth_gate_abs_mean = tanh(depth_gate).abs().mean()
```

该指标跟随现有 `logging_frequency` 记录，用于确认 gate 是否长期停在零附近。

本轮不记录完整注意力矩阵，不修改 `need_weights=False`，避免增加训练显存和计算开销。

## 5. 明确不修改的内容

本轮保持以下实现不变：

- `gate_init=0.0`；
- 同视角全局 cross-attention；
- RGB 作 Query，depth 作 Key/Value；
- 大于 `max_depth_m` 的深度视为无效；
- 三个视角共享同一套米制归一化；
- 当前一层 patch convolution 深度编码结构；
- 全无效行的 NaN 规避逻辑；
- gate 为零时 RGB checkpoint 严格等价。

## 6. 修改文件汇总

| 文件 | 修改内容 |
| --- | --- |
| `turbovla/models/depth_encoder.py` | 深度和 validity 双通道输入 |
| `third_party/starvla_runtime/starVLA/dataloader/__init__.py` | 调用 RGB-D 数据契约校验 |
| `third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/datasets.py` | 单位和相机元数据校验支持 |
| `third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py` | 关闭 center crop 并断言最终尺寸 |
| `third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py` | 记录 gate 均值 |
| `tests/test_depth_modules.py` | 双通道与 validity 测试 |
| `tests/test_turbovla_depth_forward.py` | 完整模型兼容性与 `learned_patch` 测试 |

## 7. 实施顺序

1. 修改 `MetricDepthEncoder` 为深度加 validity 双通道。
2. 补齐深度 encoder 和完整模型测试。
3. 增加单位与相机顺序校验。
4. 锁定图像处理器空间变换并增加尺寸断言。
5. 增加 `depth_gate_abs_mean` 日志。
6. 运行完整测试集。

## 8. 完成条件

只有满足以下条件才算完成：

1. 无效像素和真实中距离像素在 encoder 输入中可以区分；
2. RGB/depth token 与 invalid mask 输出形状不变；
3. 错误单位和相机顺序会在训练前抛错；
4. RGB 处理后的尺寸得到显式保证；
5. 训练日志可以观察 gate 是否开启；
6. gate 为零时 RGB-only 与 RGB-D 输出继续逐元素相等；
7. 现有及新增测试全部通过。
