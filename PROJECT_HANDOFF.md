# Tweener 火柴人插帧项目交接文档

> 更新日期：2026-08-27（Asia/Shanghai）  
> 项目目录：`/home/gezhuan/Desktop/YJH_Data/Projects/MovieType/Tweener`  
> 当前断点：v5a 高层 Python 推理接口已经通过烟雾测试；下一步实现 v5b 命令行推理入口 `infer_stickman.py`。

## 新窗口从这里开始

把下面这段话连同本文件路径发给新的 Codex 窗口：

> 请阅读 `/home/gezhuan/Desktop/YJH_Data/Projects/MovieType/Tweener/PROJECT_HANDOFF.md` 和项目根目录的 `火柴人插帧.md`，接着当前断点继续担任我的项目助理。保持原协作方式：你负责检查、解释并一步一步告诉我具体改哪个文件、把代码放在哪里、运行什么命令；不要直接替我实现项目代码。当前下一步是 v5b：设计并实现根目录的 `infer_stickman.py` 最小命令行推理工具。先检查现有文件，再只给我一个小步骤。

## 项目当前结论

项目已经建立了一条可复现的火柴人插帧研究链路：真人视频经过 MediaPipe Pose 转成 COCO-17 姿态，清洗后构造跨越 `s2/s4/s8` 的训练与独立验证序列，先后评估线性插值、残差 MLP、多帧 Transformer 和像素级 Raster U-Net。

目前正式结果支持以下结论：

- 两端姿态线性插值（LERP）是很强的基线，尤其在短间隔 `s2`。
- 残差 MLP 在 `s4/s8` 有稳定收益，但 `s2` 收益很小。
- 使用左右各 3 帧的 Transformer 在长间隔最有价值：`s8` 的姿态 MPJPE 从 LERP 的 `2.0698 px` 降到 `1.7687 px`，骨长误差从 `1.0548 px` 降到 `0.9388 px`。
- Raster U-Net 能生成更自由的像素结果，但整体拓扑稳定性不如先预测姿态再渲染。正式验证中 Transformer 的 hard Dice 在 `s2/s4/s8` 均高于 Raster last。
- v4 已验收，不应立刻重训。当前方向是把正式 Transformer 封装成可用的推理接口，再逐步扩展到任意输入和 Krita/主项目集成。

这些结果是预测与插值评测结果，不代表已经完成真实手绘火柴人、任意 PNG 或任意时间点的通用插帧。

## 协作方式（重要）

用户希望自己完成编码和运行，Codex 作为项目助理：

1. 先检查当前文件和输出，再解释结论。
2. 每次只推进一个可验证的小步骤。
3. 明确说明代码放在哪个文件、哪个位置；较大改动要给完整结构或完整文件内容。
4. 给出可以直接复制的命令，并提醒从哪个目录运行。
5. 用户执行后贴出结果，Codex 再检查并继续。
6. 除非用户明确要求，否则不要直接替用户修改项目代码。
7. 不要重复已经验收的 v0–v4，也不要因上下文切换而从头重做。



## 运行环境

从项目根目录运行：

```bash
cd /home/gezhuan/Desktop/YJH_Data/Projects/MovieType/Tweener
AI_PY=/home/gezhuan/Desktop/YJH_Data/Projects/Quant/yes/envs/ai/bin/python
```

已验证环境：

- Python：使用上面的 `AI_PY`
- OpenCV：`4.11.0`
- MediaPipe：`0.10.35`
- PyTorch：`2.5.1+cu121`
- CUDA：可用
- GPU：NVIDIA GeForce RTX 4090

所有带 `from stickman...` 的程序都应从 `Tweener` 根目录运行。曾在 `Tweener/stickman` 目录运行 `train_raster.py`，出现过 `ModuleNotFoundError: No module named 'stickman'`；这不是包代码损坏，而是工作目录错误。

## 阶段状态


| 阶段                             | 状态    | 产出与结论                                                       |
| ------------------------------ | ----- | ----------------------------------------------------------- |
| v0 合成数据与传统基线                   | 完成    | `zero_velocity`、LERP、角度插值；验证评测管线和两帧非线性歧义                    |
| v1 真人视频、姿态提取、清洗与三连帧            | 完成    | 训练/验证视频分离；COCO-17；`s2/s4/s8` 数据与基线评测                        |
| v2 残差 MLP                      | 完成    | 三个 stride 独立模型；长间隔优于 LERP                                   |
| v3 多帧 Transformer              | 完成    | 左右各 3 帧；一个模型支持 `s2/s4/s8`；正式验证完成                            |
| v4 Raster U-Net                | 完成并验收 | 公平的六帧条件、统一图像评测、案例诊断与可视化均完成                                  |
| v5a 高层 Python API              | 完成    | `stickman/interpolator.py`；正式 Transformer checkpoint 烟雾测试通过 |
| v5b 命令行推理                      | 下一步   | 创建根目录 `infer_stickman.py`，将 API 变成可重复调用的 CLI                |
| v5c 多个中间帧/任意序列                 | 未开始   | 需要先定义时间和上下文更新策略                                             |
| v5d PNG/视频输入与姿态提取              | 未开始   | 当前 API 接受 COCO-17 姿态，不直接接受任意 PNG                            |
| v5e server/backends 与 Krita 集成 | 未开始   | 当前仓库尚无 `server/backends` 目录                                 |




## 数据设计与规模



### 动作与运动类型


| 视频前缀    | 动作类型标签              |
| ------- | ------------------- |
| `walk`  | `medium_cyclic`     |
| `wave`  | `small_local`       |
| `swing` | `large_rotation`    |
| `squat` | `large_translation` |
| `lean`  | `global_rotation`   |


训练视频和验证视频为分别录制的独立视频，命名形式为 `*_train_001.mp4` 与 `*_val_001.mp4`。

### 姿态与过滤规则

- MediaPipe Pose Landmarker 输出转换为 COCO-17。
- 单关节置信度阈值：`0.35`
- 最低可信身体关节比例：`0.70`
- 画布边缘余量：`0.02`
- 最小骨长比例：`0.35`
- 最大相邻帧关节跳变：`0.12`（归一化坐标）

已知小问题：`make_real_triplets.py` 在完全缺失的相邻帧上可能出现 `All-NaN slice encountered` 警告。过滤结果仍正确，但后续可给 `np.nanmax` 增加“是否存在有限值”的保护。

### 多帧序列

上下文 `k=3`，即目标帧左侧 3 帧、右侧 3 帧，共 6 个姿态 token：

- `s2`：`[-4, -3, -2, +2, +3, +4]`
- `s4`：`[-6, -5, -4, +4, +5, +6]`
- `s8`：`[-10, -9, -8, +8, +9, +10]`

样本总数：


| 划分    | s2  | s4  | s8  | 合计   |
| ----- | --- | --- | --- | ---- |
| train | 938 | 448 | 202 | 1588 |
| val   | 698 | 336 | 154 | 1188 |


按动作细分：


| 划分    | 动作    | s2  | s4  | s8  |
| ----- | ----- | --- | --- | --- |
| train | walk  | 116 | 57  | 27  |
| train | wave  | 131 | 63  | 29  |
| train | swing | 317 | 148 | 62  |
| train | squat | 131 | 60  | 25  |
| train | lean  | 243 | 120 | 59  |
| val   | walk  | 263 | 130 | 64  |
| val   | wave  | 120 | 59  | 28  |
| val   | swing | 85  | 38  | 14  |
| val   | squat | 66  | 28  | 9   |
| val   | lean  | 164 | 81  | 39  |


注意：验证集的 `large_translation/s8` 只有 9 个样本，`large_rotation/s8` 只有 14 个样本；细分类结论的不确定性明显高于总体结果。

## 模型与正式结果



### v2：残差 MLP

结构：`68 → 128 → 128 → 34`，预测相对端点 LERP 的残差，输出层零初始化，共 `30,242` 个参数。

正式 checkpoint：

- s2：`runs/formal_mlp/mlp_s2/20260816_173845/best.pt`，epoch 25
- s4：`runs/formal_mlp/mlp_s4/20260816_173905/best.pt`，epoch 22
- s8：`runs/formal_mlp/mlp_s8/20260816_173915/best.pt`，epoch 38

验证集结果：


| stride | LERP MPJPE | MLP MPJPE | LERP 骨长误差 | MLP 骨长误差 |
| ------ | ---------- | --------- | --------- | -------- |
| s2     | 0.4782     | 0.4751    | 0.3461    | 0.3364   |
| s4     | 0.9630     | 0.9234    | 0.6206    | 0.5855   |
| s8     | 2.1675     | 1.9514    | 1.1023    | 1.0617   |


s8 骨长损失消融：`bone_weight=0` 得到 MPJPE `1.9151`、骨长误差 `1.1165`；`bone_weight=0.5` 得到 MPJPE `1.9511`、骨长误差 `1.0615`。骨长约束改善形状一致性，但略牺牲点位置误差。

### v3：多帧 Transformer

输入为 6 个 COCO-17 姿态 token 和相对时间；一个模型联合支持三个 stride。主要配置：

- `d_model=96`
- `heads=4`
- `layers=2`
- `feedforward=192`
- `dropout=0.1`
- 参数量：`166,018`
- 同样预测相对 LERP 的残差，输出层零初始化

正式 checkpoint：

`runs/formal_sequence/sequence_transformer_k3/20260816_182502/best.pt`（epoch 21）

统一验证结果：


| stride | LERP MPJPE | MLP MPJPE | Transformer MPJPE | LERP 骨长 | MLP 骨长 | Transformer 骨长 |
| ------ | ---------- | --------- | ----------------- | ------- | ------ | -------------- |
| s2     | 0.4435     | 0.4408    | 0.4459            | 0.3212  | 0.3118 | 0.3159         |
| s4     | 0.9415     | 0.9041    | 0.8836            | 0.6013  | 0.5689 | 0.5677         |
| s8     | 2.0698     | 1.8531    | 1.7687            | 1.0548  | 1.0234 | 0.9388         |


正式统一评测目录：`runs/formal_sequence_eval/20260816_183158`。

相对 LERP：

- 按验证样本自然加权，Transformer 的 MPJPE 改善 `6.79%`，骨长误差改善 `5.59%`。
- 对三个 stride 做宏平均，MPJPE 改善 `10.32%`，骨长误差改善 `7.84%`。
- 收益主要来自 `s4/s8`；`s2` 没有超过 MLP/LERP，`small_local` 类型也可能退化。



### v4：Raster U-Net

公平比较使用与 Transformer 相同的 6 帧上下文，而不是只给两个端点：

- 输入：6 张姿态前景 mask + 6 张时间图，共 `12×256×256`
- 输出：目标 mask `1×256×256`
- U-Net 通道：`16/32/64/128`
- GroupNorm + SiLU + 双线性上采样
- 在 pixel LERP 的 logit 上预测残差，输出层零初始化
- 参数量：`489,297`
- 损失：加权 BCE + Dice + `0.5 ×` 双向线条距离损失

正式训练目录：`runs/formal_raster/raster_unet_k3/20260826_164507`。

- `best.pt`：epoch 7，当前组合验证损失最低（`0.459978`）
- `last.pt`：epoch 47，组合损失较差但轮廓通常更锐利，hard Dice 更好

统一图像评测目录：`runs/formal_raster_eval/20260826_172117`。

- 验证样本：1188
- 方法：`pixel_lerp`、`pose_lerp`、`transformer`、`raster_best`、`raster_last`
- 评测行：5940
- 已检查：无重复、无缺失、无非有限指标

Hard Dice：


| 方法          | s2     | s4     | s8     | 自然加权   | stride 宏平均 |
| ----------- | ------ | ------ | ------ | ------ | ---------- |
| pixel_lerp  | 0.8192 | 0.7157 | 0.5957 | 0.7609 | 0.7102     |
| pose_lerp   | 0.9239 | 0.8606 | 0.7298 | 0.8808 | 0.8381     |
| transformer | 0.9216 | 0.8653 | 0.7577 | 0.8844 | 0.8482     |
| raster_best | 0.8678 | 0.8067 | 0.6996 | 0.8288 | 0.7914     |
| raster_last | 0.8824 | 0.8282 | 0.7268 | 0.8469 | 0.8125     |


关键配对结论：

- Transformer 相对 pose LERP：s2 平均 `-0.0023`，s4 `+0.0047`，s8 `+0.0280`。
- Raster last 相对 best：s2/s4/s8 分别 `+0.0146/+0.0215/+0.0272`，大多数样本 last 更好。
- Transformer 相对 Raster last：s2/s4/s8 分别 `+0.0392/+0.0370/+0.0309`。

视觉诊断：

- pixel LERP 常产生双轮廓和拖影。
- pose LERP/Transformer 的线条与拓扑干净，但可能出现“画得干净、姿态相位或全局位置预测错误”。
- Raster 偶尔能修正全局姿态/位置，但会出现断头、重影、肢体碎裂和拓扑错误，在 `s8 large_translation` 上尤其明显。
- `last.pt` 通常比 `best.pt` 更锐利；以后训练应同时保存 `best_total_loss.pt`、`best_hard_dice.pt` 和 `last.pt`。
- `walk_val_001_*_0496` 是稳定反例：best 优于 last，但二者整体都较差。

诊断案例：

- 选择表：`runs/formal_raster_eval/20260826_172117/selection.csv`
- 选择说明：`runs/formal_raster_eval/20260826_172117/selection.txt`
- 27 条选择记录、25 个唯一案例
- 可视化位于同目录的 `selection_grids/`，包含三个 stride 的 route/checkpoint 对比图

v4 的重要限制：所有目标图都由同一 COCO-17 渲染器生成，因此姿态坐标路线天然占有拓扑和线宽一致性的优势。这些结果不能直接外推到真实手绘线稿。

## 当前断点：v5a 已通过

新文件：`stickman/interpolator.py`

其中的 `StickmanInterpolator` 封装正式 `SequencePredictor`，目前提供：

- `epoch`、`context`、`strides`、`device` 属性
- `predict_midpoint(left_poses, right_poses, left_offsets, right_offsets)`
- `render_midpoint(...)`
- 预测形状 `(17, 2)` 和有限值检查

最近烟雾测试使用正式 Transformer checkpoint，结果：

```text
设备： cuda
epoch： 21
context： 3
支持 stride： (2, 4, 8)
预测形状： (17, 2)
全部有限： True
预测图： runs/v5_smoke/prediction.png
真值图： runs/v5_smoke/target.png
```

已确认以下文件存在：

- `stickman/interpolator.py`
- `runs/v5_smoke/prediction.png`
- `runs/v5_smoke/target.png`

这证明了“正式 checkpoint → 高层 API → COCO-17 预测 → PIL 渲染”的最短链路可用，但还不是终端用户推理工具。

## 下一步：v5b 最小命令行推理工具

下一窗口不要直接做服务器或 Krita。先在项目根目录创建 `infer_stickman.py`，将刚通过测试的 API 变成稳定、可复现的 CLI。

建议第一版只接受一个已经存在的 sequence `meta.json`，避免同时引入视频解码、姿态检测和任意时间插值。建议接口：

```bash
"$AI_PY" infer_stickman.py \
  data/val_sequences/swing_val_001/k3/s8/swing_val_001_s8_k3_0008/meta.json \
  --checkpoint runs/formal_sequence/sequence_transformer_k3/20260816_182502/best.pt \
  --device cuda \
  --out runs/v5_cli_smoke
```

建议最小职责：

1. 解析 `meta.json` 路径、checkpoint、device、输出目录和画布尺寸。
2. 复用 `SequenceDataset` 已验证的读取逻辑，或抽取一个单样本读取函数；不要复制一套不一致的归一化规则。
3. 校验样本的 `context`、stride/时间偏移是否与 checkpoint 兼容。
4. 调用 `StickmanInterpolator.render_midpoint(...)`。
5. 至少输出：
  - `prediction.png`
  - `target.png`（当 meta 有真值时，用于当前阶段核验）
  - `prediction_pose.npy` 或等价 JSON
  - `result.json`（记录 checkpoint、epoch、device、stride、offsets、输入 meta 和输出路径）
6. 在终端打印与 v5a 烟雾测试相同的关键元数据和有限值检查。

第一版验收条件：

- 从 `Tweener` 根目录执行成功。
- 指定样本能生成预测图和真值图。
- 输出姿态为 `(17, 2)` 且全部有限。
- 输出元数据能追溯到输入样本和 checkpoint。
- CLI 结果与当前 `runs/v5_smoke/prediction.png` 的同一输入推理一致（允许文件编码不同，姿态数值应一致）。
- `"$AI_PY" -m py_compile infer_stickman.py` 通过。

完成 v5b 后再决定 v5c 的接口。不要现在承诺原始的 `interpolate(prev_png, next_png, n)`：当前正式模型需要左右各 3 帧、只预测一个中心时刻，且训练只覆盖 stride `2/4/8`，两端 PNG 和任意 `n` 与模型假设不兼容。

## 后续路线（v5c–v5e）



### v5c：多个中间帧和任意序列

需要明确：

- 每个目标时刻如何选择左右 3 帧上下文。
- 非 `s2/s4/s8` 时间偏移是拒绝、量化还是重新训练支持连续时间。
- 多次递归预测是否会积累姿态漂移。
- API 是一次预测一个目标时刻，还是输入完整序列后批量补帧。



### v5d：PNG/视频输入

需要把已有 MediaPipe 提取流程改造成可调用模块，完成：

- 图片/视频帧 → MediaPipe Pose
- MediaPipe landmarks → COCO-17
- 置信度、缺失姿态和画布边界处理
- 与训练时完全一致的坐标归一化



### v5e：主项目和 Krita 集成

当前仓库没有 `server/backends`。应在稳定 CLI 和输入契约之后再设计：

- 后端类或服务 API
- 模型生命周期与 GPU/CPU 选择
- Krita 插件传入的数据格式
- 错误反馈、进度、批处理和缓存



## 关键文件索引



### 项目说明与数据处理

- `火柴人插帧.md`：主路线与阶段总结；已确认包含 v4 正式实现、统一评测和验收结论
- `extract_poses.py`：视频姿态提取和预览
- `make_real_triplets.py`：真人三连帧数据
- `make_real_sequences.py`：左右各 k 帧的序列数据
- `evaluate.py`：传统基线/MLP 评测



### 姿态模型

- `train.py`：MLP 训练
- `train_sequence.py`：Transformer 训练
- `evaluate_sequence.py`：统一姿态评测
- `visualize_sequence_eval.py`：姿态模型可视化
- `stickman/dataset.py`
- `stickman/sequence_dataset.py`
- `stickman/losses.py`
- `stickman/models/mlp.py`
- `stickman/models/predictor.py`
- `stickman/models/seq.py`
- `stickman/models/sequence_predictor.py`



### Raster 模型和诊断

- `train_raster.py`
- `evaluate_raster.py`
- `summarize_raster_eval.py`
- `select_raster_cases.py`
- `visualize_raster_checkpoint.py`
- `visualize_raster_selection.py`
- `stickman/raster_dataset.py`
- `stickman/raster_losses.py`
- `stickman/models/unet.py`



### 当前 v5

- `stickman/interpolator.py`：正式 Transformer 的高层 Python API
- `infer_stickman.py`：下一步创建，当前尚不存在
- `stickman/skeleton.py`：COCO-17 骨架定义与渲染
- `stickman/models/__init__.py`：模型导出



## 不要丢失的正式产物

- Transformer：`runs/formal_sequence/sequence_transformer_k3/20260816_182502/best.pt`
- Transformer 统一评测：`runs/formal_sequence_eval/20260816_183158`
- Raster 正式训练：`runs/formal_raster/raster_unet_k3/20260826_164507`
- Raster 正式统一评测：`runs/formal_raster_eval/20260826_172117`
- v5a 烟雾测试：`runs/v5_smoke`

在开始下一步前，建议只读核对：

```bash
cd /home/gezhuan/Desktop/YJH_Data/Projects/MovieType/Tweener

ls -lh \
  stickman/interpolator.py \
  runs/formal_sequence/sequence_transformer_k3/20260816_182502/best.pt \
  runs/v5_smoke/prediction.png \
  runs/v5_smoke/target.png

"$AI_PY" -m py_compile stickman/interpolator.py
```



## 仍可改进但目前不阻塞 v5

- 数据量仍小，且每个运动类型只有一段训练视频和一段验证视频；后续应增加人物、视角、速度和绘制风格。
- `s8` 的某些动作验证样本很少，应补充后再做强分类结论。
- 当前目标来自 COCO-17 固定渲染，不等价于真实手绘线稿。
- Transformer 在 `s2` 和部分 `small_local` 样本没有稳定超过 LERP，可考虑 stride gating 或短间隔直接回退 LERP。
- Raster 的训练选择指标与最终 hard Dice/视觉质量不一致；若重训，应拆分 checkpoint 选择标准。
- 过滤代码中的全 NaN 警告可清理。
- 目前只做中心帧预测，没有任意时间参数、序列级一致性或递归误差控制。
- 当前没有单元测试套件；v5 CLI 稳定后应补充最小测试，包括 checkpoint 兼容性、shape、finite、确定性和错误输入。



## 交接完成标准

新窗口的助理只要完成以下三项，就已经正确接上项目：

1. 阅读本文件与 `火柴人插帧.md`，确认不重做 v0–v4。
2. 检查 `stickman/interpolator.py` 和正式 Transformer checkpoint。
3. 从 v5b 的 `infer_stickman.py` 开始，一次只指导用户完成一个小步骤并等待验证。

