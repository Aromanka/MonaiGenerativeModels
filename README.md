# EHR-Conditioned OCT Latent Diffusion

本仓库在归档版 [Project-MONAI/GenerativeModels](https://github.com/Project-MONAI/GenerativeModels) 上实现了一套可恢复、可验证的微调流水线。最终模型学习：

\[
p(\mathrm{OCT\ B\mbox{-}scan}\mid z_{\mathrm{EHR}},\ \mathrm{view})
\]

其中 `z_EHR` 是已有 EHR encoder 产生的患者 latent feature。代码使用已经部署成功的 MONAI CXR LDM checkpoint：保留 2D、单通道、512×512、3-channel latent 和 1024 维 cross-attention 结构，用新的 EHR projector 替换 CLIP text encoder。

> 本项目仅用于研究，不应用于临床诊断。合成图像不能视为患者真实检查结果。

## 已实现的模型

```text
EHR latent [B, D] ── train-set normalization ──┐
                                               ├─ EHRConditionProjector
OCT view field ID ── learned embedding ────────┘         │
                                                         ▼
                                                [B, 8, 1024]
                                                         │ cross-attention
OCT ── OCT-adapted AutoencoderKL ── z ── noise ── DiffusionModelUNet
 ▲                                                       │
 └──────────────────── frozen decoder ◀──────────────────┘
```

关键设计如下：

- Autoencoder 与 Diffusion U-Net 均从官方 CXR LDM 权重严格加载；
- Autoencoder 保持原架构，先在 OCT 上做 domain adaptation；
- EHR projector 将 `[B, D]` 映射为 8 个 1024 维 context tokens；
- `cross_attention_dim=1024` 不变，因此 CXR cross-attention 权重可完整复用；
- 使用官方相同的 scaled-linear schedule 和 `v_prediction`；
- alignment 阶段只训练 projector 与 U-Net 的 `attn2` cross-attention；
- full 阶段训练 projector 与完整 U-Net，Autoencoder 始终冻结；
- 10% condition dropout、learnable null context 和 classifier-free guidance 已集成；
- diffusion checkpoint 包含 EMA、优化器、LR scheduler、GradScaler、RNG 与数据指纹，可恢复训练。

## 数据契约

默认配置在 `configs/oct_ehr_ldm.json`。

### 训练数据

`data/ukbehr_ehr_with_oct.pt` 必须是纯字典：

```python
{
    patient_id: ehr_latent,  # int -> torch.Tensor[D]，也接受 [1, D]
}
```

只有这个文件中的患者能进入 train/val。代码再与 OCT JSON 取交集，并进行确定性的患者级划分；同一患者的所有眼、所有图像永远只属于一个 split。

### 缺失 OCT 的推理数据

`data/ukbehr_ehr_only.pt` 采用同样格式，但绝不会被训练或验证代码读取。它只供最终 `sample` 命令生成缺失模态。

### OCT 图像索引

服务器索引默认为：

```text
/data/home/wanglidi/code/encode_oct/data/OCT_eid.json
```

格式来自 `encoder/README.md`：

```json
{
  "1791781": [
    "/absolute/path/1791781_21018_0_0_image88605_64.png",
    "/absolute/path/1791781_21017_0_0_image88604_64.png"
  ]
}
```

当前图像是已经抽取的中央 B-scan（`image*_64`），因此 slice position 对所有样本是常量，没有加入伪造的 slice-position embedding。文件名第二段 UKB field ID（示例中的 `21017`、`21018`）作为通用 `view_code` 条件；代码不擅自将其命名为左眼或右眼。无法解析的文件名使用 `<unknown>`。

图像会转为单通道 `[0,1]`，并默认按原宽高比缩放、黑边补齐到 512×512。没有使用翻转、90° 旋转或 CutMix 等可能破坏眼部解剖/双眼语义的增强；默认只有 3% 的轻微强度扰动。

## 环境

第一阶段部署已完成的环境中执行：

```bash
cd /path/to/MonaiGenerativeModels

pip install -e . --no-deps
pip install -r requirements-oct.txt
```

`requirements-oct.txt` 中的 `lpips` 用于 Autoencoder perceptual loss。若服务器不能安装或下载其权重，可将配置中的：

```json
"perceptual_weight": 0.0
```

此时使用 L1 + KL 完成最小微调。建议继续使用已验收环境中的固定 PyTorch、MONAI 与 CUDA 版本，不在本阶段升级核心依赖。

确认两个原始 checkpoint 存在：

```text
model-zoo/models/cxr_image_synthesis_latent_diffusion_model/models/autoencoder.pth
model-zoo/models/cxr_image_synthesis_latent_diffusion_model/models/diffusion_model.pth
```

## 服务器执行顺序

所有命令都从仓库根目录运行。全局 `--config` 要放在子命令之前。

### 1. 校验数据并固化患者级 split

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json prepare-data
```

该命令会检查：

- `.pt` 顶层确实是 `patient_id -> Tensor`；
- 所有 EHR latent 都是一致的有限 `[D]` 向量；
- EHR 与 OCT JSON 有交集；
- JSON 中图像文件真实存在；
- train/val 患者无重叠。

输出为：

```text
outputs/data/split_manifest.json
```

manifest 包含固定的 patient IDs、图像记录、view mapping 和数据指纹。后续阶段复用这个文件，避免不同阶段意外重新划分。查看摘要：

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json inspect-data
```

如果数据更新，应先归档旧 checkpoint，再显式删除旧 manifest 并重新执行 `prepare-data`。checkpoint 会拒绝与不同数据指纹混用。

### 2. 服务器 smoke test

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json smoke-test
```

它读取一个真实 batch，并对原始 Autoencoder 与 Diffusion U-Net 执行 `strict=True` 加载。此命令不会开始训练。

### 3. Stage 0：评估原始 CXR Autoencoder

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  evaluate-autoencoder \
  --checkpoint model-zoo/models/cxr_image_synthesis_latent_diffusion_model/models/autoencoder.pth \
  --output-dir outputs/reconstruction_cxr_init \
  --max-panels 16
```

输出 `metrics.json` 及若干横向 panel：

```text
original | reconstruction | absolute error
```

重点观察视网膜层边界、RPE、高反射层、囊腔/积液等小结构是否丢失。

### 4. Stage 1：OCT Autoencoder domain adaptation

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json train-autoencoder
```

输出：

```text
outputs/autoencoder/best.pt
outputs/autoencoder/last.pt
outputs/autoencoder/metrics.jsonl
```

然后重新评估：

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  evaluate-autoencoder \
  --checkpoint outputs/autoencoder/best.pt \
  --output-dir outputs/reconstruction_oct_adapted \
  --max-panels 32
```

配置中的 `paths.oct_autoencoder_checkpoint` 已默认指向这个 `best.pt`。只有当 OCT-adapted reconstruction 明显优于 CXR 初始化，并保留关键视网膜结构时，才进入 diffusion。

恢复 Autoencoder：

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  train-autoencoder --resume outputs/autoencoder/last.pt
```

### 5. Stage 2：EHR condition alignment

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  train-diffusion --phase alignment
```

首次运行会在训练集上重新估计 OCT-adapted latent 的标准差，并使用：

```text
scale_factor = 1 / std(z_OCT)
```

该值写入 checkpoint；不会错误沿用 CXR sampler 的固定 `0.3`。alignment 阶段只训练：

```text
EHRConditionProjector + DiffusionModelUNet.*.attn2.*
```

输出：

```text
outputs/diffusion_alignment/best.pt
outputs/diffusion_alignment/last.pt
outputs/diffusion_alignment/metrics.jsonl
```

恢复同一阶段：

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  train-diffusion --phase alignment \
  --resume outputs/diffusion_alignment/last.pt
```

### 6. 检验模型是否使用 EHR condition

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  evaluate-diffusion \
  --checkpoint outputs/diffusion_alignment/best.pt \
  --max-batches 20
```

报告三种固定噪声下的 v-prediction loss：

- `correct_loss`：正确患者 EHR；
- `shuffled_loss`：替换为另一个 val 患者的 EHR，但保留当前 view；
- `null_loss`：learned null condition；
- `condition_gap = shuffled_loss - correct_loss`。

理想情况下 `condition_gap > 0`。它不是最终图像质量指标，但能发现 projector/condition 被完全忽略的问题。若 val 只有一位患者，无法定义可靠的 shuffled 指标，应增大 val split。

### 7. Stage 3：完整 Diffusion U-Net 微调

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  train-diffusion --phase full \
  --init-checkpoint outputs/diffusion_alignment/best.pt
```

这是一个新阶段：默认把 alignment checkpoint 的 EMA diffusion/projector 权重作为初始化，然后训练完整 U-Net 与 projector。不要把 `--init-checkpoint` 当作恢复；中断后应改用：

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  train-diffusion --phase full \
  --resume outputs/diffusion_full/last.pt
```

完成后再次运行条件评估：

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  evaluate-diffusion \
  --checkpoint outputs/diffusion_full/best.pt
```

## 使用 EHR-only 患者生成 OCT

先为一位缺失 OCT 的患者生成两个已训练 view 的图像：

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  sample \
  --checkpoint outputs/diffusion_full/best.pt \
  --patient-id 1234567 \
  --samples-per-view 4 \
  --guidance-scale 4.0 \
  --inference-steps 50 \
  --output-dir outputs/generated/eid_1234567
```

`--ehr-file` 未提供时默认读取 `data/ukbehr_ehr_only.pt`。未提供 `--view-code` 时，对 checkpoint 中每个已训练 field ID 都生成；也可显式指定：

```bash
--view-code 21017 --view-code 21018
```

批量生成所有 EHR-only 患者必须显式使用 `--all`，防止误触发大任务：

```bash
python -m oct_ehr_ldm --config configs/oct_ehr_ldm.json \
  sample \
  --checkpoint outputs/diffusion_full/best.pt \
  --all \
  --output-dir outputs/generated/ehr_only_all
```

默认使用 EMA 权重；调试原始权重可加 `--no-ema`。每个输出目录还会保存 `samples.json`，记录 patient ID、view、seed、CFG scale 和步数。

## 默认训练参数与显存

默认值面向单张 24 GB GPU：

| 项目 | 默认值 |
|---|---:|
| resolution | 512×512 |
| device batch | 2 |
| gradient accumulation | 4 |
| precision | 自动选择 BF16，否则 FP16 |
| condition tokens | 8×1024 |
| condition dropout | 0.1 |
| alignment steps | 10,000 |
| full steps | 100,000 |
| EMA | 0.9999 |
| gradient clip | 1.0 |

显存不足时按顺序调整：

1. `data.batch_size: 2 -> 1`；
2. `gradient_accumulation_steps: 4 -> 8`，维持 effective batch；
3. `autoencoder.activation_checkpointing: true`（仅影响 Autoencoder 阶段）；
4. 缩短验证 batch 数或降低 DataLoader workers 不会显著降低训练显存，但可减少主机内存/I/O 压力。

EMA 会额外保存一份模型参数。若 24 GB 环境仍无法容纳 full 阶段，可临时设置 `diffusion.use_ema=false`，但最终生成质量通常更适合使用 EMA，推荐优先减小 device batch。

## 输出和验收标准

每个阶段的 `metrics.jsonl` 是逐行 JSON，可直接被 pandas、SwanLab、W&B 或自定义脚本读取。至少完成以下验收：

```text
[ ] prepare-data 显示 train/val patient 无交集
[ ] 原始 CXR 两个 checkpoint strict=True 加载
[ ] OCT-adapted Autoencoder 的 MAE/PSNR 和视觉结构优于 CXR 初始化
[ ] latent scale factor 为有限正数并写入 diffusion checkpoint
[ ] alignment/full loss 无 NaN，gradient norm 有限
[ ] correct EHR loss 优于 shuffled/null（至少在足够大的 val 上呈趋势）
[ ] 固定 seed 可重复生成，改变 EHR 时输出有可测的条件变化
[ ] 最终使用独立患者做图像质量与临床特征一致性评估
```

单纯 diffusion loss 不能证明图像真实或 EHR 一致。正式实验还应在患者级 test split 上报告生成质量（例如 FID/KID 或医学特征空间距离）、多样性、RETFound 特征距离，以及与 EHR 相关的疾病/生物标志物一致性；评估 encoder 不能只复用训练目标而没有独立验证。

## 常见问题

### `.pt` 不是预期纯字典

`prepare-data` 会给出实际类型。若服务器文件带有包装层，例如：

```python
{"features_by_eid": {...}, "metadata": {...}}
```

不要静默猜测字段；先转换为明确的 `{int: Tensor[D]}` 文件，或按真实契约修改 `load_ehr_dictionary()`。

### JSON 中存在失效绝对路径

默认 `fail_missing_images=true` 会立即失败并列出示例。确认服务器挂载路径，或重新运行 `encoder/OCT_extract_middle_image.py`。不建议在正式训练中跳过未审计的大量缺失图像。

### `lpips` 导入或权重失败

先执行 `pip install -r requirements-oct.txt`。仅为了跑通最小实验时，可将 `perceptual_weight` 设为 0；要在实验记录中注明 loss 改动。

### full 阶段应该使用 `--init-checkpoint` 还是 `--resume`

- alignment → full：`--init-checkpoint outputs/diffusion_alignment/best.pt`；
- full 中断继续：`--resume outputs/diffusion_full/last.pt`；
- 二者不能同时使用。

### 为什么不直接使用 `ehr_only.pt` 训练

这些患者没有配对 OCT，无法构造监督的 conditional diffusion target。它们的用途是训练完成后做缺失模态生成，而不是混入 paired training。
