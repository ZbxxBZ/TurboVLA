# TurboVLA 改进与实验结果

本文记录当前 TurboVLA RGB-D 改进方案、两阶段训练流程、使用的数据以及已经得到的训练结果。

## 1. 改进目标

原始 TurboVLA 使用 DINOv3 提取多视角 RGB token，再经过 BERT 条件化的 VisionLanguageInteraction（VLI）和 ACT 动作解码器生成动作。当前改进只将 `cam_head`（头部相机）的 RGB 用于几何分支：

1. 用 VGGT 的预训练几何 Transformer 提取头部 RGB 的几何特征。
2. 复用 VGGT 官方 DPT depth head，把中间特征融合为 dense feature，并用 RoboTwin 真实深度做像素级监督。
3. 将监督后的 DPT 融合 feature 池化为 `14 x 14 = 196` 个深度 token。
4. 在动作训练阶段，让经过 VLI 语言条件化的 RGB token 通过 gated cross-attention 查询深度 token。

腕部相机仍参与原始 RGB policy 输入，但当前深度分支只对头部相机生效；腕部对应的深度 token 被 mask 掉，不参与深度残差注入。

## 2. 模型结构

### 2.1 RGB 与语言主干

```text
三路 RGB（cam_head、left_wrist、right_wrist）
        |
        +--> DINOv3 RGB encoder（Stage 2 冻结）
        |        |
        |        +--> RGB tokens
        |
        +--> BERT instruction encoder（Stage 2 冻结）
                 |
                 +--> text tokens

RGB tokens + text tokens
        |
        +--> VisionLanguageInteraction（6 层，Stage 2 冻结）
                 |
                 +--> 语言条件化 RGB tokens
```

### 2.2 深度分支

```text
cam_head RGB [B, 3, 224, 224]
        |
        +--> 还原 ImageNet normalization，缩放到 518 x 518
        |
        +--> VGGT Aggregator
        |        （VGGT 多层 Transformer 几何主干）
        |
        +--> VGGT 官方 DPTHead
        |        |
        |        +--> depth output：预测深度图
        |        +--> depth_conf：预测置信度图
        |        +--> fused feature F：[B, 256, Hf, Wf]
        |
        +--> F 自适应池化到 14 x 14
                 |
                 +--> flatten + transpose
                          |
                          +--> depth tokens [B, 196, 256]
```

在 Stage 1 的 `dpt_dense` 模式中，DPTHead 的 fused feature 直接作为深度 token 的输入表示，因其通道数已经是 256，所以不再增加额外的 160 -> 256 token projection。深度 token 仅保留头部视角；为了与三视角 policy 接口兼容，其他视角使用全 invalid mask。

### 2.3 深度融合与动作输出

```text
语言条件化 RGB tokens [B, 3, 196, 256]
深度 tokens           [B, 3, 196, 256]
        |
        +--> GatedDepthCrossAttention
        |       Query = RGB tokens
        |       Key/Value = 对应视角的全部深度 tokens
        |
        +--> delta
        |
        +--> gate * delta
        |
        +--> fused RGB tokens = RGB tokens + gate * delta
        |
        +--> 冻结的 ACT decoder
                 |
                 +--> 未来 50 步、14 维动作
```

对于头部视角，RGB token 可以查询全部有效深度 token，而不是只能读取同一空间位置的深度 token。cross-attention 输出的残差为 `delta`，融合形式为：

```text
X_fused = X_rgb + g * delta
```

其中 `g` 是 256 维逐通道 gate。当前配方使用 `tanh` 参数化，`gate_init=0.0`，因此第 0 步严格保持 RGB-only 输出；训练过程中 gate 不再使用外部 warmup 强制值。cross-attention 残差会先按 RGB token 的 RMS 做幅度对齐，使 `depth_gate_rms` 成为有效视角 residual ratio 的直接对照；逐通道 `abs_mean` 只作为辅助统计。

## 3. Stage 1：VGGT-DPT dense 深度适配

### 3.1 训练内容

Stage 1 的目标是把 VGGT 的通用几何特征适配到 RoboTwin 的 `cam_head` RGB-D 分布，使 DPTHead 输出的深度和 RoboTwin 传感器深度一致。训练时：

- 冻结 VGGT Aggregator，即 VGGT 的 Transformer 几何主干。
- 解冻并训练 VGGT 官方 `depth_head` / DPTHead。
- 训练可学习的 metric scale 与 shift，用于将 VGGT 的深度尺度校准到米制深度。
- 不训练 DINOv3、BERT、VLI 或 ACT；这些模块不属于 Stage 1 的深度适配任务。

### 3.2 监督信号

数据集中的深度以毫米保存，读入后除以 1000 转为米，并使用最近邻插值到 `224 x 224`。监督是 dense pixel-level supervision，不是每个 patch 只使用一个平均深度值。

对有效像素（有限值且在配置的深度范围内）计算：

```text
L_stage1 = scale-invariant log-depth loss
           + 0.05 * local gradient loss
```

- scale-invariant log-depth loss 约束预测深度与真实深度的相对深浅关系，并降低整体尺度偏移的影响。
- local gradient loss 约束深度边界和局部结构，例如物体边缘、桌面边界。
- 无效深度像素被 mask，不参与损失和指标计算。

DPTHead 同时产生 `depth`、`depth_conf` 和 fused feature。训练损失使用 `depth` 与真实 dense depth 比较；`depth_conf` 是 VGGT 输出的置信度，不需要额外的人工置信度标签。fused feature 在 Stage 2 中作为深度 token 来源。

### 3.3 Stage 1 数据与配置

- 数据来源：RoboTwin `cam_head` RGB-D 采集数据。
- 轨迹数：360 条。
- 帧数：65,515 帧。
- 任务数：50 个 Clean 任务。
- 相机：只读取 `vision/cam_head/colors` 和 `vision/cam_head/depths`。
- 划分：按 episode 划分训练集和验证集，默认验证比例为 10%。
- batch size：8。
- 训练 epoch：5。
- VGGT 输入：单帧头部 RGB，缩放到 `518 x 518`。

Stage 1 的 dense 训练脚本保留了 `legacy_patch` 选项，但本次结果使用的是 `stage1_mode=dpt_dense`。`legacy_patch` 只作为兼容旧实验的可切换路径，不是本次 Stage 1 结果的来源。

### 3.4 Stage 1 验证结果

指标定义：

- `loss`：验证集上的 dense log-depth 损失。
- `Abs Rel`：所有有效像素的 `|pred - gt| / gt` 平均值。
- `MAE`：所有有效像素的绝对深度误差平均值。
- `RMSE`：所有有效像素的平方误差均方根。

服务器日志 `/root/logs/vggt_dpt_dense_stage1_5ep_20260827.log` 的结果如下：

| Epoch | Validation loss | Abs Rel | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.02543 | 0.01052 | 6.64 mm | 16.94 mm |
| 2 | 0.02247 | 0.00740 | 4.53 mm | 15.03 mm |
| 3 | 0.02209 | 0.00820 | 4.88 mm | 14.80 mm |
| 4 | **0.02114** | **0.00678** | **4.07 mm** | **14.18 mm** |
| 5 | 0.02138 | 0.00761 | 4.67 mm | 14.28 mm |

第 4 epoch 的 `Abs Rel` 最低，因此对应 checkpoint 是 Stage 1 的最佳验证结果。最终用于 Stage 2 的 Stage 1 adapter 为：

```text
/root/checkpoints/stage1_dpt_dense_20260827/best.pt
```

Stage 1 结果表明 DPT dense 分支已经拟合了 RoboTwin 的深度尺度和局部结构；第 5 epoch 没有继续改善，因此 Stage 1 继续增加 epoch 的收益有限。

## 4. Stage 2：动作深度融合训练

### 4.1 初始化与冻结策略

Stage 2 从官方 TurboVLA RGB checkpoint 初始化，并加载 Stage 1 的 DPT adapter：

```text
官方 checkpoint：/root/models/steps_55000_ema_model.safetensors
Stage 1 adapter：/root/checkpoints/stage1_dpt_dense_20260827/best.pt
```

启动日志确认官方 checkpoint 恢复了 878 个 RGB 模型张量，并初始化了新增的深度张量。Stage 2 的参数策略为：

| 模块 | 状态 |
|---|---|
| RGB DINOv3 | 冻结 |
| BERT | 冻结 |
| VGGT Aggregator | 冻结 |
| Stage 1 DPTHead / adapter | 冻结 |
| VisionLanguageInteraction | 冻结 |
| ACT/action decoder | 冻结 |
| `depth_fusion.cross_attention` | **训练** |
| `depth_fusion.depth_gate` | **训练** |
| `depth_fusion.depth_norm` | **训练** |

总参数约 1,376.068M，可训练参数约 0.263M。因此本阶段主要学习“如何把已经得到的深度表示注入现有 RGB policy”，而不是重新学习视觉、语言或动作能力。

### 4.2 Stage 2 训练配置

- 数据：`/root/datasets/robotwin_lerobot_clean50_360`。
- 轨迹/帧数：360 条轨迹、65,515 帧。
- 任务：50 个 RoboTwin Clean 任务。
- policy 训练读取 RGB、状态和动作；真实深度只在 Stage 1 使用，Stage 2 的深度 token 由头部 RGB 经冻结 VGGT-DPT 分支得到。
- batch：`per_device_batch_size=16`，`gradient_accumulation_steps=1`，有效 batch size=16。
- 总优化步数：32,760，约 4,095 步/epoch，共 8 epoch。
- 优化器：AdamW；cross-attention/depth-fusion 学习率为 `1e-4`。
- gate：`tanh(gate_raw)`，从 0 开始学习；没有 `depth_gate_warmup` 覆盖。
- 残差监控：`depth_gate_rms` 与 `depth_residual_ratio_valid_*`；all-view 指标保留为明确标注的诊断值。
- mixed precision：BF16。
- DataLoader：8 workers、pin memory、persistent workers、prefetch factor=4。
- EMA：衰减率 `0.999`，保存普通模型和 EMA 模型。
- checkpoint：每 4,095 步保存一次，因此每个 epoch 都保存一份普通 checkpoint 和一份 EMA checkpoint。

本次从 `8/2` 切换为 `16/1` 后，有效 batch size 没变。两种设置的优化目标相同，但由于 batch 内聚合顺序、BF16 舍入和随机性，loss 不会逐位完全一致；16/1 冒烟测试在 3090 24 GB 上两步成功，峰值显存约 15.6 GiB。

### 4.3 Stage 2 已完成 epoch 的结果

下面这组数值来自此前的 `gate=0.15` bounded-sigmoid/warmup 运行，作为历史基线保留；它不代表本次新配方（`tanh gate=0`、幅度对齐、可训练 `depth_norm`）的结果。新配方需要单独启动后再记录对应日志。

下表使用每个 epoch 边界附近的最近日志点（日志每 50 step 记录一次，因此使用 step 4100、8200 等，而不是不存在的精确 4095 日志点）。`action loss` 是该记录点的动作 loss，不是整 epoch 的算术平均值；它用于观察趋势。

| 已完成 Epoch | 代表 Step | action loss | gate mean | depth residual ratio allviews mean |
|---:|---:|---:|---:|---:|
| 1 | 4,100 | 0.07794 | 0.15000 | 0.902 |
| 2 | 8,200 | 0.05369 | 0.15052 | 1.009 |
| 3 | 12,300 | 0.06501 | 0.15084 | 1.048 |
| 4 | 16,400 | 0.05131 | 0.15109 | 1.169 |
| 5 | 20,500 | 0.04973 | 0.15130 | 1.133 |
| 6 | 24,600 | **0.04258** | 0.15150 | 1.175 |

趋势解读：

- action loss 从第 1 epoch 边界的约 `0.078` 降到第 6 epoch 边界的约 `0.043`，总体下降，期间有正常的 batch 波动。
- gate 从固定的 `0.15` 缓慢变化到约 `0.1515`，没有塌缩到下界，也没有冲到上界。
- `depth residual ratio mean` 约从 `0.90` 增长到 `1.18`，说明 cross-attention 输出的深度残差已经对 RGB token 产生了非零且有一定幅度的影响。该比例不是越大越好，最终仍需通过 RoboTwin 成功率和正确深度/打乱深度消融来判断深度是否带来真实任务收益。

截至本记录生成时，Stage 2 已完整完成 6 个 epoch，第 7 个 epoch 正在进行；已生成的 checkpoint 为：

```text
steps_4095
steps_8190
steps_12285
steps_16380
steps_20475
steps_24570
```

每个 step 目录同时包含普通模型和 EMA 模型。当前运行目录为：

```text
/root/turbovla_stage2_runs/turbovla_robotwin_clean50_360_depth_stage2_8ep_b16a1_20260828_hz4
```

## 5. 当前结论与后续验证

目前可以确认：

1. Stage 1 的 DPT dense 分支已经在 RoboTwin RGB-D 数据上得到较低的像素级深度误差。
2. Stage 2 只更新约 0.263M 个深度融合参数，动作 loss 在前 6 个 epoch 总体下降。
3. gate 没有回到 0；同时 residual ratio 明确非零，说明深度分支没有被完全忽略。
4. 这些训练指标不能直接等同于任务成功率。下一步应使用 Stage 2 的 EMA checkpoint，在与官方 checkpoint 相同的 RoboTwin 任务和随机种子下评测，并至少增加以下消融：
   - 正确的深度 token；
   - batch 内打乱的深度 token；
   - 深度分支置零。

只有正确深度明显优于打乱/置零，并且 click 类任务成功率提升，才能说明深度分支学到了可用于控制的几何信息。
