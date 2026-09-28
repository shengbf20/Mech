# README
>实验计划

**先用 4 张卡复现 Gemma-2-2B 上的 IMDb rehearsal priming**

这里的“复现成功”指：**在 Stage 3 起点的情感分数已经匹配后，有 Stage 1 经历的模型再次学习正面影评时，曲线稳定快于首次学习对照**。IMDb 是便于测量的代理任务，不能单凭它得出安全对齐的结论。

## 1. 需要写哪些代码

建议做成以下几个小程序，共用一份配置文件：

| 程序 | 职责 | 必须留下的结果 |
|---|---|---|
| `prepare_data.py` | 下载后固定数据版本和随机种子；清理文本；划分正负训练集、校准集、最终测试集；按 Gemma tokenizer 处理长度 | 数据 ID、样本数、哈希、划分文件 |
| `train_sft.py` | 同一套全参数 SFT 代码运行 Alpaca 预热及 Stage 1/2/3；支持从**模型权重**开始新阶段并重置优化器 | 各阶段权重、步数、loss、配置 |
| `generate_eval.py` | 从固定的 IMDb 前 2–8 个 token 继续生成；统一采样参数和随机种子 | 每个检查点逐题生成的文本 |
| `score_eval.py` | 用固定的 RoBERTa 分类器计算正面比例（**仅 `positive` 计入，`neutral` 不计**）；抽样检查分类错误 | 逐题标签、汇总分数 |
| `match_checkpoint.py` | 在**校准集**上寻找 Stage 2 分数接近共同起点的检查点 | 每条轨迹选中的检查点及分数差 |
| `analyze.py` | 画三阶段曲线；比较 Stage 3 前期增量、达到同一目标分数所需步数；按题目和训练种子计算区间 | 图、表和可复核的原始结果 |
| `config.yaml`、`ds_zero2.json` | 固定模型、数据版本、全局 batch、学习率、精度、保存与评测间隔 | 一次实验可完整重跑 |

训练采用 TRL 的 `SFTTrainer`、AdamW 和 DeepSpeed ZeRO-2。**不要依赖 TRL 对损失范围的默认判断**：预热与 IMDb 的文本格式、loss 范围见 **§1.2**（均为我们的实现选择）。论文给出了影评数量和评测前缀方式，但没有把 IMDb 的训练文本格式细化到足以逐字重建。[TRL SFT 文档](https://huggingface.co/docs/trl/sft_trainer)、[DeepSpeed ZeRO 文档](https://www.deepspeed.ai/tutorials/zero/)

> **含义限定：**「确定」仅指**论文写明了什么 / 我们已锁定写什么**；不能保证照做一定观察到 priming 效应；也不能从「TRL 默认」反推作者当时的精确配置。

## 1.0 已确认的研究决策

已按推荐方案落实以下 5 点，后续实验与结论表述以此为准：

1. **样本设计：**主实验复用同一组 8192 条正面影评；本轮同时做样本互斥对照（每侧 **6144** 条），并在对照内部保持样本量和训练步数一致。
2. **任务范围：**首阶段先用 IMDb 观察 rehearsal priming，之后再考虑 BeaverTails。
3. **结论边界：**只陈述是否观察到行为层面的再次学习加速，不预设结果，也不据此证明机制。
4. **Stage 2 匹配：**若评测波动过大，按预先锁定的预算增加采样（默认 1 条/题，细化时 **5** 条/题）；仍无法可靠满足匹配条件的分支不纳入主比较。
5. **分析指标：**以正式实验前锁定窗口上、相对基线调整后的曲线面积为主指标，达到目标分数的步数为辅助指标；先以 **128 步**作为试跑窗口。

## 1.1 第一轮参数（论文可对齐 / 已拍板的协议）

下列写入 `config.yaml` 后不得在实验中途更改。**不含**由「4 卡 × 每卡 4」推导出的全局 batch、bf16、预热轮数等——那些属 §1.2。

- **模型与预处理：**`Gemma-2-2B` **base**（非 `-it`）；实验前用 `Alpaca-cleaned` 预热。性质：论文设定（预热**是否做**；**做多久**见 §1.2）。
- **训练：**全参数 SFT、TRL、DeepSpeed ZeRO-2、AdamW、恒定 LR \(5\times10^{-7}\)、weight decay `0`、最大序列长度 `512`。性质：论文设定。
- **硬件与每卡 batch：**原实验 `4×A800 80GB`，**每卡 batch 4**。性质：论文设定；**未写** `gradient_accumulation_steps`。
- **阶段数据规模：**每阶段随机抽取 `8192` 条；IMDb 使用正、负影评。性质：论文设定。
- **生成及分数：**采样 temperature `1.0`；IMDb 影评开头 `2–8` 个 token 作提示；**仅 `positive` 计入正面比例**，`neutral`/`negative` 记 0。性质：temperature 与前缀范围属论文；计分映射为我们锁定的协议。
- **起点匹配：**Stage 2 每 `128` 步粗查接近 \(S_0\) 的区间，再逐步评测；\(\lvert S-S_0\rvert<0.005\)。性质：论文协议。点估计过线≠起点等价；校准报告须含逐题差异与重复采样波动。波动处置与不合格分支剔除见 **§1.0 第 4 点**。

## 1.2 第二轮参数（我们的实现选择；**已全部确认并写入 `config.yaml`**）

> **硬约束：**任何实验代码若需**修改**本节已锁定值，必须先向负责人报备；不得擅自改 `config.yaml`。

下列均为**复现声明用的实现选择**（不可标成作者原配置）。与 `config.yaml` 不一致时以仓库内已锁定文件为准。

- **全局 batch：****已锁定。**`gradient_accumulation_steps: 1`、`global_batch_size: 16`（4 卡 × 每卡 4）。**不是**论文直接报告的数值。8 卡时每卡改为 2。
- **IMDb 输入与 loss：****已锁定。**原始影评、无聊天/指令包装；BOS + 文末 EOS；除 BOS、padding 外对影评 token 算 loss。正式训练前打印并人工检查 10 条 token/label。
- **Alpaca 预热：****已锁定。**固定 Alpaca 风格模板（见 `config.yaml`）；只对 `output` 算 loss；**1 epoch**。
- **Stage 1/3 样本关系：****已锁定（§1.0）。**主实验复用 8192 正面；互斥对照每侧 **6144** 条（全局 batch 16 → 384 步），对照内样本量与步数一致。
- **评测划分：****已锁定。**`test` 上互不重叠的 **2000 校准 + 2000 最终测试**，正负平衡。
- **前缀：****已锁定。**Gemma tokenizer；题目 ID 哈希 → 长度 `2…8`；全检查点共用。
- **生成：****已锁定。**`do_sample=true, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=128`，遇 EOS 停。默认每题 1 条。
- **分类器：****已锁定仓库。**`j-hartmann/sentiment-roberta-large-english-3-classes`（revision 见 config）；`positive→1`，其余→0。通路阶段用有标签 IMDb 核对表现。
- **\(S_0\)：****已锁定定义。**Alpaca 预热后共同起点在校准题上的分数；保存权重与逐题生成。
- **优化细项：****已锁定。**`betas=(0.9,0.999)`、`eps=1e-8`、`warmup_steps=0`、`max_grad_norm=1.0`、`packing=false`；各阶段重置优化器。
- **精度与显存：****已锁定。**`bf16=true`；A800/A6000 均开 gradient checkpointing。
- **种子：****已锁定。**`[11, 23, 37]`。
- **Stage 1 深度：****候选已写入。**`128/256/512`；仅可按校准试跑改档，正式三种子前终锁。
- **Stage 3 评测间隔：****已锁定（可于通路后微调并再报备）。**前 128 步每 8 步，之后每 32 步。
- **主分析指标：****已锁定（§1.0）。**基线调整后曲线面积为主；达目标分数步数为辅；试跑窗口 128 步。正式窗口与目标分数试跑后终锁。
- **匹配加采样预算：****已锁定。**默认每题 1 条；当 \(\lvert S-S_0\rvert<0.02\) 进入细化时每题增至 **5** 条；仍不可靠则该分支不入主比较。
- **版本钉：****HF revision 与当前环境库版本已写入 `config.yaml`。**通路成功后另冻 `requirements.lock.txt`；实验中途不升级。

## 2. 需要联网准备什么

| 资源 | 建议下载项 | 用途与注意点 |
|---|---|---|
| 基础模型及 tokenizer | [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)，**base 版，不是 `-it`** | 下载前需登录 Hugging Face 并接受 Gemma 使用条款 |
| 指令预热数据 | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | 形成所有分支共用的预热起点；论文未明确预热步数，须自行固定并报告 |
| 主实验与评测数据 | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) | 只从 `train` 取训练影评；从 `test` 固定抽取校准和最终测试提示 |
| 情感评测模型 | [j-hartmann/sentiment-roberta-large-english-3-classes](https://huggingface.co/j-hartmann/sentiment-roberta-large-english-3-classes) | 与论文引用的 Hartmann 情感分类器相符的**实现候选**；论文没有给出精确模型仓库 ID，须如实记录这一差异 |
| 软件 | PyTorch/CUDA、Transformers、TRL、Datasets、Accelerate、DeepSpeed、绘图及统计库 | 固定一套实际能运行的版本和依赖锁文件，之后不要在实验中途升级 |

下载模型和数据时固定 **revision/commit**，不要只记录仓库名。先用 IMDb 已标注的完整影评检查分类器在该领域是否工作正常，再用它评价生成内容。IMDb 官方划分为 25,000 条训练、25,000 条测试。[IMDb 数据卡](https://huggingface.co/datasets/stanfordnlp/imdb)

## 3. 两种机器怎么用

论文报告 **4×A800 80GB、每卡 batch 4、最大长度 512、全参数微调、恒定学习率 \(5\times10^{-7}\)**。**未写明**梯度累积与全局 batch。已锁定（§1.2 / `config.yaml`）：`gradient_accumulation_steps=1` → 全局 batch 16：

- **4×A800 80GB：**每卡 batch 4，贴近论文每卡设定。
- **8×A800：**若全用 8 张，每卡 batch 2，保持全局 batch 16；更简单的首轮方案是只用其中 4 张。先确认是 80GB 还是 40GB。
- **4×RTX A6000 48GB：**按已锁定配置试每卡 batch 4、长度 512、bf16、ZeRO-2 与梯度检查点。若显存不足，改为每卡 batch 2、梯度累积 2，**仍保持全局 batch 16**（硬件适配，须记录并报备改 config）。

**4×A6000 做 2B 全参数训练有较大可行性**，但不能仅凭显存标称保证每卡 batch 4 一定装得下。先跑 20–50 步实测峰值显存、每步时间及保存/恢复。ZeRO-2 检查点中的优化器状态会明显大于纯权重。

## 4. 可落地的执行顺序

**第 0 步：锁定实验定义。****已完成。**§1.0 / §1.1 / §1.2 已确认；机器可读配置见 `config.yaml` 与 `ds_zero2.json`。[论文附录](https://arxiv.org/html/2605.18309)

**第 1 步：下载、划分并检查数据。****已完成。**产物在 `data/splits/`（`manifest.json`）；互斥 Stage1/3 overlap = 0。

**第 2 步：做最小通路测试。****已完成。**4 卡 ZeRO-2 训练 32 步 → 保存 → 重载 → 生成 → RoBERTa 打分。指标见 `outputs/pathway/`（约 2.6 s/step，本 rank 峰值显存约 18 GB）；依赖已冻为 `requirements.lock.txt`。
```
deepspeed --num_gpus=4 pathway_smoke.py --stage train --max-steps 32
CUDA_VISIBLE_DEVICES=0 python pathway_smoke.py --stage eval
```

**第 3 步：预热并冻结共同起点。****已完成。**Alpaca 1 epoch → `outputs/warmup/W0`（3235 步）；校准集 2000 题生成并打分，\(S_0=\) **`positive_rate_S = 0.261`**（见 `outputs/warmup/S0_calib_scored.summary.json`）。后续分支均从该 \(W_0\) 出发。
```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
deepspeed --num_gpus=4 train_sft.py --mode warmup --config config.yaml
CUDA_VISIBLE_DEVICES=0 python generate_eval.py \
  --config config.yaml \
  --model outputs/warmup/W0 \
  --prompts data/splits/calib_2000.jsonl \
  --output outputs/warmup/S0_calib_generations.jsonl
CUDA_VISIBLE_DEVICES=0 python score_eval.py \
  --config config.yaml \
  --generations outputs/warmup/S0_calib_generations.jsonl \
  --output outputs/warmup/S0_calib_scored.jsonl
```

**第 4 步：Stage 1 正面训练。****已完成（种子 11 试跑）。**从 \(W_0\) 起在 `main_positive_8192` 上训满 512 步（≈45 min，本 rank 峰值显存 ≈17.8 GB），并在 128/256/512 落盘；校准集 2000 题、4 卡并行生成后打分如下（\(S_0=0.261\)）：

| Stage 1 深度 | 校准 `positive_rate_S` | \(\Delta S_0\) |
|---|---:|---:|
| 128 | **0.292** | +0.031 |
| 256 | **0.345** | +0.084 |
| 512 | **0.313** | +0.052 |

**深度锁定：**未见过早饱和（128 仅小幅抬升；三者均远未贴顶），正式主分支锁定 **Stage 1 = 512 步**（`outputs/stage1/main_pos_seed11/depth_512`）。256→512 非单调，记为单次采样波动，不据此改浅；三种子正式跑前若复现再议。产物：`outputs/stage1/main_pos_seed11/`（`train_metrics.json`、`depth_*`、`calib_scored_d*.summary.json`）。
```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 4a. 训练（种子 11 试跑；优化器从 W0 权重新建，不恢复旧态）
CUDA_VISIBLE_DEVICES=4,5,6,7 \
deepspeed --num_gpus=4 train_sft.py --mode imdb \
  --config config.yaml \
  --model outputs/warmup/W0 \
  --data data/splits/main_positive_8192.jsonl \
  --output outputs/stage1/main_pos_seed11 \
  --max-steps 512 \
  --save-depths 128,256,512 \
  --seed 11

# 4b. 各深度在校准集上生成并打分（判断是否过早饱和；正式深度锁定前只看 calib）
# generate：4 卡按题号分片并行（i % 4 == rank）；score 仍单卡即可
for D in 128 256 512; do
  CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_eval.py \
    --config config.yaml \
    --model outputs/stage1/main_pos_seed11/depth_${D} \
    --prompts data/splits/calib_2000.jsonl \
    --output outputs/stage1/main_pos_seed11/calib_gen_d${D}.jsonl \
    --num-gpus 4
  CUDA_VISIBLE_DEVICES=7 python score_eval.py \
    --config config.yaml \
    --generations outputs/stage1/main_pos_seed11/calib_gen_d${D}.jsonl \
    --output outputs/stage1/main_pos_seed11/calib_scored_d${D}.jsonl
done
```

**第 5 步：Stage 2 反向训练与匹配。****已完成（种子 11）。**从 `depth_512` 出发在 `main_negative_8192` 上反向 SFT；粗/中/细匹配后选中 **`depth_34`**，校准集 n=5 得分 \(S=0.2638\)，\(|S-S_0|=0.0028<0.005\)（\(S_0=0.261\)）。产物：`outputs/stage2/main_neg_seed11/MATCHED.json`、`depth_34/`。

```bash
# 已跑通的一键命令（可续跑）
GPUS=4,5,6,7; SCORE_GPU=7
python stage2_run.py --phase all \
  --stage1 outputs/stage1/main_pos_seed11/depth_512 \
  --data data/splits/main_negative_8192.jsonl \
  --output outputs/stage2/main_neg_seed11 \
  --seed 11 --max-steps 512 \
  --train-gpus $GPUS --gen-gpus $GPUS --score-gpu $SCORE_GPU --num-gpus 4
```

**第 6 步：Stage 3 同步比较。****已完成（种子 11 试跑）。**两臂同数据 `main_positive_8192`、同种子 11、窗口 128 步（每 8 步评 + step 0）：
- **primed**：`outputs/stage2/main_neg_seed11/depth_34`
- **control**：`outputs/warmup/W0`

产物：`outputs/stage3/seed11/curve_report.json`，分臂曲线 `eval/{primed,control}/curve.json`。

| step | \(S_{\mathrm{primed}}\) | \(S_{\mathrm{control}}\) | \(\Delta\) |
|---:|---:|---:|---:|
| 0 | 0.259 | 0.261 | −0.002 |
| 24 | 0.292 | 0.244 | +0.048 |
| 40 | 0.342 | 0.242 | +0.101 |
| 64 | 0.350 | 0.254 | +0.096 |
| 96 | 0.336 | 0.300 | +0.036 |
| 128 | 0.321 | 0.298 | +0.023 |

主指标（梯形面积，窗口 \([0,128]\)）：

| 指标 | 值 |
|---|---:|
| `area_delta_S`（primed−control） | **+6.972** |
| primed 相对自身 \(S(0)\) 面积 | +8.234 |
| control 相对自身 \(S(0)\) 面积 | +1.006 |

```bash
cd /home/u-shengbf/mech
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
GPUS=4,5,6,7
SCORE_GPU=7
python stage3_run.py --phase all \
  --seed 11 \
  --primed outputs/stage2/main_neg_seed11/depth_34 \
  --control outputs/warmup/W0 \
  --data data/splits/main_positive_8192.jsonl \
  --output outputs/stage3/seed11 \
  --max-steps 128 \
  --train-gpus $GPUS --gen-gpus $GPUS --score-gpu $SCORE_GPU --num-gpus 4
```

**第 7 步：重复并出结论。****暂缓（先不进入）。**原计划种子 `[11,23,37]` 与互斥对照；当前仅完成种子 11 试跑，三种子与互斥臂留待后续。

## 5. 实验结论（种子 11 试跑；截至 Stage 3）

> **结论边界（§1.0）：**只陈述是否观察到行为层面的再次学习加速；不据此证明机制；单种子、校准集 n=1 采样，不作正式三种子推断。

### 5.1 流程是否跑通
**是。**通路 → Alpaca 预热 \(W_0\)（\(S_0=0.261\)）→ Stage 1 正面（锁深 512）→ Stage 2 负面并匹配到 \(S_0\)（`depth_34`，n=5：\(S=0.2638\)，\(|S-S_0|=0.0028\)）→ Stage 3 双臂同步再学正面，曲线与主指标已落盘。

### 5.2 是否观察到 rehearsal priming（行为层）
**是（本试跑支持）。**在起点几乎对齐（primed \(S(0)=0.259\)，control \(S(0)=0.261\)）的前提下：
- primed 在约 16–40 步内快速抬升并在 ~0.33–0.35 平台；
- control 前几十步几乎不涨甚至略降，约 72 步后才缓慢接近 ~0.30；
- 窗口内 `area_delta_S = +6.97 > 0`，primed 相对自身起点的曲线面积约为 control 的约 **8×**。

这与「有 Stage 1 经历的模型，在分数回到共同起点后再学正面时，曲线更快/更高」的行为现象一致。

### 5.3 不声称什么
- 不声称已验证论文中的 Rebound / Driving Force 或后验印记机制；
- 不声称结果在种子 `[23,37]` 或互斥样本对照上稳健（第 7 步未做）；
- 不从 IMDb 代理任务外推到安全对齐。

### 5.4 当前状态
**暂停于第 6 步完成态；第 7 步（多种子 + 互斥对照）暂缓。**可复核原始结果目录：`outputs/warmup/`、`outputs/stage1/main_pos_seed11/`、`outputs/stage2/main_neg_seed11/`、`outputs/stage3/seed11/`。

这条路线主要依据你提供的 :codex-file-citation{path="D:\sbf\0 NJU课程资料\Creations\Alignment Dynamics\Alignment Dynamics in LLM Fine-Tuning.pdf" purpose="source"}。它能检验论文报告的**行为现象**；即使观察成功，也还需要额外的表示或梯度实验才能验证论文提出的具体机制。