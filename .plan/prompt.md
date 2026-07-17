现在我需要你创建一个数据生成脚本。其将会根据给出的ehr数据，生成对应的OCT图像。

# 生成模块
生成模块的内核是（样例指令）：
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

# 输入数据格式
输入数据的读取方式为：
```python
import pickle

with open("Dataset/generated/ukb_train_trajectories.pkl", "rb") as f:
    payload = pickle.load(f)

trajectory = payload["trajectories"][1000010]
print(trajectory["tokens"])
print(trajectory["ages_days"])
print(trajectory["cutoff_age_years"])
print(trajectory["ehr_latent"])
```

# 输出数据格式
输出数据需要整理成如下格式，以便接入其他项目：
## 目标结论

最终交付给训练框架的文件应为原生 Schema V2，它包含预编码的 EHR 与 OCT 特征，而不是原始图片。原始 OCT 图片和 manifest 作为审计材料保留。

Schema 定义见 [schema.py](</E:/code/project/xdiabetes/dtmh/xdiabetes2/src/data/schema.py:178>)，模态字段定义见 [base.py](</E:/code/project/xdiabetes/dtmh/xdiabetes2/src/models/encoders/base.py:59>)。

## 推荐交付目录

```text
Dataset/synthetic/ukbehr_ehr_oct/
├── images/
│   ├── 1000010/
│   │   ├── 1000010_synthetic_0000_right.png
│   │   └── 1000010_synthetic_0000_left.png
│   └── ...
├── oct_manifest.csv
├── OCT_features_synthetic.pt
├── dataset_manifest.json
└── ukb_synthetic_train.pt       # 最终训练输入
```

`oct_manifest.csv` 建议包含：

```text
patient_id,visit_id,index_time_days,image_name,image_path,laterality,generator_checkpoint,seed,qc_pass
1000010,1000010@synthetic_0000,26750.0,1000010_synthetic_0000_right.png,images/1000010/...,right,...,42,true
```
对于patient_id，你可以从1开始重新编码。只要保证所有数据条目（patient）编码唯一，train/val split不会重合即可。
## OCT 特征中间格式
生成的 OCT 图片必须使用与真实 UKB 数据完全相同的 RETFound checkpoint 和预处理流程编码。当前真实缓存的 OCT 维度是 `1024`，每位患者有1个 OCT token。
```python
{
    "features_by_eid": {
        1000010: {
            "image_names": [
                "1000010_synthetic_0000_right.png",
                "1000010_synthetic_0000_left.png",
            ],
            "features": torch.FloatTensor[N_oct, 1024],
        },
    },
    "metadata": {
        "synthetic": True,
        "feature_dim": 1024,
        "encoder_name": "RETFound",
        "encoder_checkpoint": "...",
        "generator_name": "...",
        "generator_checkpoint": "...",
    },
}
```

## 最终 Schema V2 格式

顶层结构：

```python
{
    "schema_version": 2,
    "samples": [
        {
            "patient_id": str,
            "visit_id": str,
            "index_time": float,
            "modalities": {
                "ehr": {...},
                "oct": {...},
            },
            "target": {...},
            "metadata": {...},
        },
        ...
    ],
    "meta": {...},
}
```

单条样本的精确规范：

| 字段 | 格式 | 说明 |
|---|---|---|
| `patient_id` | `str` | 原始 EHR trajectory 对应的 UKB EID |
| `visit_id` | `str` | 唯一状态 ID，如 `1000010@synthetic_0000` |
| `index_time` | `float` | EHR trajectory 的 `cutoff_age_days` |
| EHR features | `[1,120] float32` | EHR 生成结果中的 `ehr_latent` |
| OCT features | `[N_oct,1024] float32` | 生成图片经过 RETFound 后的特征 |
| target value | `[6] float32` 全零 | 仅作为占位符 |
| target mask | `[6] bool` 全假 | 表示完全未标注 |

推荐使用项目 API 构造，避免手写 Schema：

```python
import torch

from src.data.schema import EncodedSample, build_v2_payload
from src.models.encoders.base import EncoderOutput

LABEL_NAMES = [
    "token_220",
    "token_221",
    "token_222",
    "token_223",
    "token_224",
    "token_961",
]

patient_id = "1000010"
visit_id = f"{patient_id}@synthetic_0000"
index_time = float(ehr_trajectory["cutoff_age_days"])

ehr_feature = torch.as_tensor(
    ehr_trajectory["ehr_latent"],
    dtype=torch.float32,
).reshape(1, 1, 120)

oct_feature = torch.as_tensor(
    oct_record["features"],
    dtype=torch.float32,
).reshape(1, -1, 1024)

num_oct_tokens = oct_feature.shape[1]

sample = EncodedSample(
    patient_id=patient_id,
    visit_id=visit_id,
    index_time=index_time,
    modalities={
        "ehr": EncoderOutput(
            features=ehr_feature,                    # (1, 1, 120)
            token_mask=torch.ones(1, 1, dtype=torch.bool),
            timestamps=torch.tensor([[index_time]]),
            observed_mask=torch.tensor([True]),
            quality=torch.ones(1, 1),
            fidelity=torch.ones(1, 1),
            generated=torch.ones(1, 1, dtype=torch.bool),
            metadata={
                "trajectory_length": len(ehr_trajectory["tokens"]),
                "cutoff_age_days": index_time,
            },
            provenance={
                "source": "synthetic_ehr_trajectory",
                "encoder": "DelphiSMURF",
            },
        ),
        "oct": EncoderOutput(
            features=oct_feature,                    # (1, N_oct, 1024)
            token_mask=torch.ones(
                1, num_oct_tokens, dtype=torch.bool
            ),
            timestamps=torch.full(
                (1, num_oct_tokens), index_time
            ),
            observed_mask=torch.tensor([True]),
            quality=torch.ones(1, num_oct_tokens),
            fidelity=torch.ones(1, num_oct_tokens),
            generated=torch.ones(
                1, num_oct_tokens, dtype=torch.bool
            ),
            metadata={
                "image_names": oct_record["image_names"],
                "synthetic": True,
            },
            provenance={
                "source": "synthetic_oct",
                "encoder": "RETFound",
                "generator_checkpoint": "...",
                "encoder_checkpoint": "...",
            },
        ),
    },
    target={
        "case_weight": torch.tensor(1.0),
        "tasks": {
            "diabetes": {
                "value": torch.zeros(6, dtype=torch.float32),
                "mask": torch.zeros(6, dtype=torch.bool),
            },
        },
    },
    metadata={
        "synthetic": True,
        "unlabeled": True,
        "source_patient_id": patient_id,
    },
)
```

对于OCT影像编码，请参考子项目（子文件夹）中的 @encoder\README.md 。由于OCT生成，OCT编码，最终数据整理这三个阶段独立性较高，你可以分为多个脚本，由一个bash脚本统一流程。你的所有脚本都创建在 @factory 文件夹下。