# RGB-D 交叉注意力分支评审与修改计划（v2）

评审对象：在 main 分支 RGB-only 基线之上引入 RoboTwin 真实深度，让 RGB token 作 Query、深度 token 作 Key/Value 做门控交叉注意力的这套改动。

涉及文件：

- `turbovla/models/depth_encoder.py`、`turbovla/models/depth_fusion.py`（新增模块）
- `turbovla/models/turbovla.py`、`turbovla/models/configuration.py`（接入与校验）
- `third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py`（batch 组装、初始化加载）
- `third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/{datasets,schema,depth_io}.py`（数据侧）
- `third_party/starvla_runtime/starVLA/training/trainer_utils/trainer_tools.py`（学习率分组）
- `experiments/robotwin/evaluation/model2robotwin_interface.py`（推理侧）
- `experiments/robotwin/configs/clean50_depth.yaml`、`experiments/robotwin/configs/modality_depth.json`

v2 说明：v1 的若干结论经 `DEPTH_FUSION_REVIEW_FEEDBACK.md` 复核后被推翻或降级，本版已修正。第 1 节明确列出撤回的结论，避免错误判断继续影响后续决策。除已注明的两处代码验证外，本文仍是静态评审。

## 1. v1 中已撤回的结论

| v1 结论 | 判定 | 依据 |
| --- | --- | --- |
| gate < 约 `4e-3` 时 bf16 舍入使前向逐位不变 | **错误，撤回** | 残差加法在 fp32 下完成，见 1.1 |
| 位置参数从预训练 RGB checkpoint 加载，与主干争夺同一参数 | **错误，撤回** | 已自行验证，见 1.2 |
| log 归一化下 `0.05-0.5m` 占约 55% 动态范围，head camera 有用区间被压扁 | **算错，撤回** | 实为 50%，见 1.3 |
| 深度编码器"没有非线性、表达不了边缘" | **表述错误，撤回** | 见 1.4 |
| post-fusion LayerNorm 是"更便宜的保险" | **错误，撤回** | 会破坏 gate=0 等价性，见 1.5 |
| 给 `delta` 乘 `1/sqrt(d)` | **无依据，撤回** | 缺少实验支撑 |
| center crop 会"悄悄失效" | **严重性降级** | 存在两道保护，见 3.2 |

### 1.1 bf16 死区不存在

v1 的推理是：`fused = rgb + gate * delta` 中 rgb 为 O(1)，bf16 相对精度约 `2^-8`，故 gate 小于约 `4e-3` 时加法被舍入吃掉。

这条链路不成立。`VisionProjection` 的最后一层是 `output_norm`（LayerNorm），autocast 下 LayerNorm 在 fp32 执行并输出 fp32；`delta` 出自 MultiheadAttention 为 bf16。`fp32 + bf16` 按类型提升在 **fp32** 下相加，因此不存在固定舍入门槛。反馈给出的探针结果（gate=`1e-4` 时最大绝对变化约 `8.5e-5`，约 99.93% 元素改变）与此一致。

保留的部分：`gate_init=0.0` 时第 0 步只有 `depth_gate` 获得非零梯度，深度编码器与交叉注意力参数梯度严格为零。这是零初始化残差门控的预期行为，`tests/test_depth_modules.py:43` 已覆盖。它会自行解除（gate 一旦离开零点，梯度即进入深度分支），因此**不再建议**把默认 `gate_init` 改成 0.05–0.1。

### 1.2 位置参数不从初始化 checkpoint 加载（已验证）

`_load_initialization`（`TurboVLA.py:227-259`）的映射表只有四条前缀：`bert.`、`feat_map.`、`transformer.encoder.text_layers.`、`transformer.encoder.fusion_layers.`。`patch_position_embedding`、`patch_position_scale`、`view_embedding` 均不在其中，是当前模型新建的参数。v1 所述"与预训练分布冲突"不成立。

顺带确认了一个相关疑问：这三个参数是 `TurboVLA` 的顶层参数，不属于 `lr_cfg` 里任何具名模块。`build_param_lr_groups`（`trainer_tools.py:102-106`）有 `other_params` 兜底组，它们以 `base=5e-5` 训练，不存在"被排除在优化器之外"的问题。

因此 v1 的"待确认前置数据：打印 checkpoint 里 `patch_position_scale` 的值"这一项作废——该参数总是从 `position_scale_init=0.01` 开始。

### 1.3 log 范围分配的算术更正

正确值为：

```text
ln(0.5/0.05) / ln(5/0.05) = ln(10)/ln(100) = 50%   （不是 55%）
ln(2.0/0.8) / ln(5/0.05) = 19.9%                    （head camera 主要工作区间）
线性归一化下同一区间: (2.0-0.8)/(5.0-0.05) = 24.2%
```

即 log 相对线性只让 `0.8-2.0m` 少拿约 4 个百分点。一个 2.5 倍深度跨度占到近 20% 的动态范围并不算被压扁，v1 结论撤回。共享米制归一化还保留了一条物理先验：同一归一化值在三个相机中代表同一真实距离。分视角 min/max 会破坏该语义，不应作为首选。

### 1.4 编码器容量的表述更正

v1 称编码器"没有非线性、表达不了边缘"，不准确：可选 log 变换本身是非线性的，卷积后还有 LayerNorm；而线性卷积完全可以表达边缘与梯度（Sobel 就是线性算子）；`16x16 -> 256` 的输入输出维度相同，原则上不丢 patch 内信息。

站得住的限制只有：patch 之间没有编码器级上下文；非重叠 patch 对跨边界结构不敏感；validity 语义歧义；跨模态匹配完全从随机初始化开始学。

### 1.5 post-fusion LayerNorm 会破坏等价性

这是反馈中一个有价值的更正。LayerNorm 无条件归一化，因此 gate=0 时 `LN(rgb) != rgb`，与 RGB-only checkpoint 的严格等价性立即失效，而这条等价性正是整个分支设计的兼容性基础（`tests/test_turbovla_depth_forward.py:124` 依赖它）。v1 把它称为"更便宜的保险"是错的。

## 2. 对反馈本身的两处反向修正

反馈整体成立，但有两处判断我认为需要反过来调整。

### 2.1 P3 应当被改写，而不只是降级为"待测量"

反馈第 7.3 节指出：RGB Query 来自 DINOv3，本身携带位置与全局上下文，随机初始化的 Q/K 投影也会让不同 Query 得到不同权重，因此不能由"位置项小"推出"注意力近似均匀"。这一点正确，v1 的"退化成全图池化摘要"确实没有证据。

但被否证的是一个非核心命题。核心问题不是注意力是否均匀，而是**能否建立对角（同位置）对应**，而这里存在一个结构性、无需测量即可确认的不对称：

- Q 侧：DINOv3 输出的 token 内容本身就是位置相关的（位置信息在主干内部已被编码进内容），量级 O(1)。
- K 侧：`patch_embed` 是 stride 等于 kernel 的卷积，**平移等变**。两块深度内容相同的 patch，无论位于图像何处，产生的 token 完全相同。深度 key 的位置身份**只能**来自那个约 `1e-4` 的加性位置项。

也就是说，深度 key 在初始化时近似位置无关，Q 侧再丰富的位置信息也无法匹配到不编码位置的 key 上。可检验的预测因此不是"注意力均匀"，而是**"注意力非均匀但空间不对齐"**——按内容相似度聚集到图像各处深度相近的 patch 上。这比均匀更难通过看熵发现，因为熵会显示为正常的低熵。

所以诊断指标不能只看注意力熵：必须同时看对角质量与局部窗口质量（反馈第 7.3 节的指标清单里已包含这两项，方向是对的）。相应地，v1 的建议"给深度分支一套独立、量级与内容可比的位置编码"依然是这套设计里最直接的对症改动，只是应该按反馈要求进入消融而非直接改。

反馈提出的"全局注意力也有优点（容忍 RGB-D 小幅错位、利用对应位置之外的有效深度、获得物体级上下文）"我同意，这也正是"局部/全局混合"应作为消融项之一的理由。

### 2.2 语义分离是反馈自己所提消融的前置条件，不能推迟

反馈把 P1.2（`>5m` 视为无效）判定为设计取舍，并建议先统计再消融，其中一项消融是"`>5m` 屏蔽 vs clamp 后保留"。

问题在于：当前代码**无法表达后者**。`depth_encoder.py:57` 的 `valid` 把 `isfinite`、`>= min`、`<= max` 三个条件与在一起，"缺失"和"太远"共用同一个 flag，随后一起进入 token 级 `invalid_threshold` 判定。要跑这个消融，必须先把两种语义拆开。

因此"把缺失与超远分离"不是一项待定的架构修改，而是**任何相关消融的前置代码改动**，且它本身不改变默认行为（默认仍可把超远计入无效）。同理适用于 P1.1：要比较"有/无 validity 通道"，得先有能表达 validity 的输入路径。

由此得出一条对反馈阶段划分的补充意见，见第 5 节：`patch_embed` 的 `in_channels` 应在长训练**之前**定下来，否则第二阶段产出的 checkpoint 无法作为第四阶段的热启动。

## 3. 应立即处理的防误配项

三项共同点：失败时不抛异常、训练照常收敛，只是策略变差。均为高价值断言，不代表当前数据已经错误。

### 3.1 数据集深度单位与模型配置缺少闭环校验

`datasets.py` 把 `depth_unit` 写入 simplified metadata，`MetricDepthEncoder` 只按 `config.depth.input_unit` 决定是否除以 `depth_scale`，二者之间没有校验。

若数据声明为米而模型配置仍为毫米，`0.5-5.0m` 会被当成 `0.0005-0.005m`，全部低于 `min_depth_m=0.05` 并进入无效 mask，训练继续跑但深度残差长期为零。

反馈确认当前官方链路不存在该误配（RoboTwin `camera.py` 乘 1000 存 uint16 毫米、附加脚本声明 `millimeter`、评估接口也转 uint16 毫米），因此定位为防误配断言。

### 3.2 显式锁定图像处理器的空间变换

`TurboVLA.py:136-137` 只设置了 `image_processor.size`，未设置 `crop_size`、未关闭 `do_center_crop`，预处理配置不封闭。LIBERO 那条路径是显式关掉的（`turbovla/data/libero_rlds.py:106`），此处应保持一致。

严重性按反馈意见降级：`size` 被强制为 `224x224`，若 `crop_size` 更小会改变 patch 数，`learned_patch` 下 `_position_visual_tokens` 会因位置长度不匹配报错，`view` 下 `GatedDepthCrossAttention` 也会因 RGB/depth 形状不等报错；`crop_size` 更大则无法裁剪。因此"必然静默错位"未被证实，准确表述是：空间变换依赖外部处理器配置，需显式锁定并加最终尺寸断言。

本机未设置实际 `DINOV3_MODEL_PATH`，无法读取训练所用本地处理器的 `crop_size`/`do_center_crop` 最终值，该项需在训练机上确认。

### 3.3 RGB 与深度相机顺序应显式校验

`data_config.py` 的 `depth_keys` 与 `modality_depth.json` 的 video 顺序当前一致，eval 也按 `[head, left, right]` 组装，但模型收到的只是两个有序张量，无法识别左右腕是否被交换。

采纳反馈的改进：比较**规范化后的完整相机 ID**并校验列表长度，不要只比字符串尾部（容易碰撞）。

## 4. 需要证据支持的表征问题

以下四项由确定性修复降级为实验假设，但其中的**语义与形状前置改动**按第 2.2 节的理由仍应先做。

### 4.1 部分无效 patch 缺少 validity 语义

`depth_encoder.py:74-75` 把归一化后的无效像素填为 `0.0`，而 `0.0` 正是有效区间的几何中点（当前配置下约 `0.5m`）。当 patch 无效比例低于 `invalid_threshold=0.5` 时该 token 仍进入 K/V，卷积无法区分 `0.0` 是"未知"还是"真实约 0.5m"。这是当前最明确的表征歧义，反馈也认定成立。

候选方案（反馈提供，需消融）：validity 通道；patch 有效比例作为附加特征；mask-aware pooling/convolution；更严格的 token 屏蔽阈值。

前置：先统计三相机的像素级与 patch 级缺失率。

### 4.2 "缺失"与"超远"应在代码层分离

见第 2.2 节。RoboTwin 相机近远平面为 `0.1m/100m`，大于 5 米的有限值可以是真实观测；而对操作任务，主动屏蔽远背景也可能有益（减少无关 K/V）。两种策略都合理，但当前代码只能表达其中一种。

改动内容：把 `valid` 拆成"缺失/非有限"与"超出范围"两个判定，默认行为保持不变（超远仍计入无效），使消融只需切换开关。

前置统计：各相机零值与非有限值比例、`>5m` 像素与 patch 比例、被屏蔽区域是否含任务相关物体。

### 4.3 归一化范围：保持共享作为基线

按第 1.3 节，v1 的压缩论证不成立。共享米制归一化作为基线保留。若统计显示三相机分布差异确实很大，优先研究 view-conditioned affine 或 view-conditioned 编码器，而不是直接给每个视角配不同物理范围——后者破坏"同一归一化值等于同一真实距离"这条先验。

### 4.4 位置编码与编码器容量

按第 2.1 节，P3 的可检验预测是"注意力非均匀但空间不对齐"，诊断需同时包含对角质量与局部窗口质量，不能只看熵。

按第 1.4 节，编码器的确定性限制只有跨 patch 上下文缺失与 validity 歧义两项。以下全部作为消融项，不作为训练前必改：独立且量级可比的深度位置编码；局部窗口注意力；对角残差路径 `delta = attn(...) + W · depth_token_i`；14x14 网格相对位置 bias；非线性 MLP 或卷积 stem；跨 depth token self-attention；深度梯度/法向输入通道。

## 5. 工程与可维护性项

成立且各自独立：

- `DepthEncoderConfig` 默认 `224/3 views` 与 `VisionEncoderConfig` 默认 `256/2 views` 冲突，仅启用默认深度配置即抛错。
- image/patch 一致性检查在 DINOv3 构建**之后**（`turbovla.py:122-127`），失败偏晚。注意反馈的限制条件：patch size 来自实际 DINOv3 配置，若要提前校验必须给 vision config 增加显式 patch size 或先只加载轻量配置，不能把现有检查原样搬进 dataclass。
- 调用方（`turbovla.py:217-221`）与 fusion 内部（`depth_fusion.py:48-51,66-69`）重复 `.to(...)`。
- `flat_depth.clone()` 每次前向分配 `B·V·196·256`，是性能问题而非正确性问题，是否优化由显存与吞吐 profiling 决定。
- `train_robotwin_clean_act_pi05_recipe.py` 默认 `--config_yaml` 从 `clean50.yaml` 改为 `clean50_depth.yaml`，改变了不带参数时的行为，属入口兼容性决策。
- 现有完整前向测试用默认 `position_embedding="view"`，未覆盖 RoboTwin 实际使用的 `learned_patch` 路径。

补充一条对反馈阶段划分的修正：`patch_embed` 的 `in_channels` 决定权重形状，一旦第二阶段的长训练用 `in_channels=1` 产出 checkpoint，第四阶段改成 2 通道就无法热启动。建议在第一阶段就把 `in_channels` 定为 2、第二通道恒为 1（validity 全有效）作为对照，这样 4.1 的消融变成数据侧改动而非权重形状改动，与反馈"先取证据"的原则不冲突。

## 6. 分阶段执行计划

### 第一阶段：低风险加固（不改变默认数值行为）

1. 校验数据集声明单位与模型 `input_unit`（3.1）。
2. 校验 RGB/depth 规范化相机 ID、数量与顺序（3.3）。
3. 显式锁定处理器 resize/crop 行为并断言最终 `H x W`（3.2）。
4. 拆分"缺失"与"超远"两个判定，默认行为不变（4.2）。
5. 把 `patch_embed` 的 `in_channels` 定为 2，第二通道恒 1（第 5 节）。
6. 补 `learned_patch` 下的 RGB-D 完整前向测试。
7. 保留显式 `gate_init=0` 的 RGB 等价性测试。
8. 回滚 CLI 默认配方为 `clean50.yaml`，深度配方由 `scripts/robotwin/train.sh` 显式传。

### 第二阶段：取得数据与训练证据

1. 三相机深度直方图：零值、非有限值、`>5m` 的像素级与 patch 级比例。
2. 像素级与 patch 级无效比例分布。
3. 记录 `tanh(depth_gate)` 的均值/最大值/分位数、深度分支梯度范数、`delta` 范数。
4. 记录注意力熵、对角质量、局部窗口质量、以及不同 RGB Query 间注意力分布的差异（对应 2.1 的"非均匀但不对齐"预测）。
5. 小数据集玩具过拟合，确认深度分支能被真正启用。

### 第三阶段：最小消融

1. validity 通道有/无（第二通道恒 1 vs 真实 mask）。
2. `gate_init=0` vs 小非零值，不预设 bf16 门槛。
3. 当前全局注意力 vs 局部窗口 vs 对角残差+全局 vs 独立位置编码。
4. `>5m` 屏蔽 vs clamp 后保留。
5. 共享归一化 vs view-conditioned 方案。

### 第四阶段：有证据后再增加容量

仅当注意力诊断与任务指标表明深度 K/V 表征不足时，再考虑非线性 MLP 或卷积 stem、跨 depth token self-attention、相对位置 bias、深度梯度/法向等几何通道。

## 7. 修订后的总体判断

工程层面依然可靠：零门控保证旧 RGB checkpoint 逐元素等价、可选模态向后兼容、深度走无损 PNG、每视角只查自己相机的 K/V、全 mask 行的 NaN 规避、独立 modality 文件、学习率组跳过 None 模块。

风险判断改写为：当前额外 patch 位置项初始幅度约 `1e-4`，且深度 key 因平移等变卷积而近似位置无关，融合模块又缺少显式局部/对角几何先验，因此初始化时能否建立稳定的 RGB-D 空间对应存在不确定性。可检验的预测是注意力非均匀但空间不对齐，需通过对角质量、局部窗口质量与短程训练实验确认，不能仅凭注意力熵判断。

零门控使第 0 步只有 gate 获得非零梯度，这是预期的兼容性设计；当前 autocast 路径下不存在固定的 bf16 前向死区，是否改用非零初始化应由 gate/梯度监控与短程消融决定。

编码器能表达 patch 内的线性几何模式，确定性限制是缺少跨 patch 上下文与 validity 语义歧义；是否增加容量由诊断与消融决定。

最重要的执行原则（采纳反馈）：不要在没有统计与短程消融之前同时改动 validity、超远语义、gate 初始化、位置编码与编码器结构，否则无法归因性能变化。第一阶段只做不改变默认数值行为的加固与形状预留。
