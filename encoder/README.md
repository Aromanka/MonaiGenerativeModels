# RETFound OCT Encoding Workflow

This folder contains the scripts needed to collect target OCT images from UK Biobank OCT zip files and encode them into patient-level RETFound feature tensors.

The workflow has two stages:

1. Extract the selected OCT image for each target patient and write an `eid -> image paths` JSON index.
2. Encode those images with RETFound and save an `eid -> image names + feature tensors` `.pt` file.

## 1. Collect Target OCT Images

Use `OCT_extract_middle_image.py` to find OCT zip files for the target patient eids, extract the selected middle OCT image from each zip, copy it into one folder, and write a JSON index.

Example:

```bash
python OCT_extract_middle_image.py \
  --zip_names_path data/oct_zip_filenames.txt \
  --target_eids_path data/train_diabetes_labels.csv \
  --OCT_basepath /data/home/UKB/data/OCT/ \
  --final_data_folder data/OCT_Images \
  --output_json data/OCT_eid.json
```

Inputs:

- `--zip_names_path`: text file containing OCT zip filenames, one per line.
- `--target_eids_path`: CSV or text file containing target patient eids. The script accepts columns named `eid`, `csvpatient_id`, or `patient_id`; otherwise it uses the first column.
- `--OCT_basepath`: directory containing the OCT zip files.
- `--final_data_folder`: directory where the selected OCT images will be copied.
- `--output_json`: output JSON mapping each patient eid to copied OCT image paths.

The resulting `data/OCT_eid.json` looks like:

```json
{
  "1791781": [
    "/data/home/wanglidi/code/encode_oct/data/OCT_Images/1791781_21018_0_0_image88605_64.png",
    "/data/home/wanglidi/code/encode_oct/data/OCT_Images/1791781_21017_0_0_image88604_64.png"
  ],
  "1472882": [
    "/data/home/wanglidi/code/encode_oct/data/OCT_Images/1472882_21018_1_0_image682970_64.png",
    "/data/home/wanglidi/code/encode_oct/data/OCT_Images/1472882_21017_1_0_image682969_64.png"
  ]
}
```

Each key is a patient eid. Each value is the list of selected OCT image paths for that patient.

## 2. Encode OCT Images

Use `encode.py` to batch encode all images listed in `data/OCT_eid.json` with the RETFound OCT checkpoint.

Example:

```bash
python encode.py \
  --input_json data/OCT_eid.json \
  --checkpoint_path /data/home/wanglidi/model/RETFound_oct_weights.pth \
  --output_pt data/OCT_features.pt \
  --batch_size 64 \
  --num_workers 4
```

Arguments:

- `--input_json`: JSON from the extraction step. Default: `data/OCT_eid.json`.
- `--checkpoint_path`: RETFound OCT checkpoint path. This argument is required.
- `--output_pt`: output feature file. Default: `data/OCT_features.pt`.
- `--batch_size`: number of images encoded per forward pass. Reduce this if GPU memory is insufficient.
- `--num_workers`: number of DataLoader workers. Set to `0` if multiprocessing causes server issues.
- `--device`: torch device, for example `cuda`, `cuda:0`, or `cpu`. Defaults to CUDA when available.
- `--image_size`: model input size. Default is `224`, matching RETFound ViT-Large patch16.
- `--no_skip_bad_images`: fail immediately on missing or unreadable images. By default, bad images are skipped.

During encoding, each image is:

1. Opened with PIL and converted to RGB.
2. Resized to `224 x 224`.
3. Converted to a tensor.
4. Normalized with ImageNet mean/std.
5. Passed through `models_vit.vit_large_patch16(num_classes=0, global_pool=True)`.

The expected feature dimension is `1024` per image.

## 3. Encoded Output Format

The output `.pt` file is saved with `torch.save` and has this structure:

```python
{
    "features_by_eid": {
        "1791781": {
            "image_names": [
                "1791781_21018_0_0_image88605_64",
                "1791781_21017_0_0_image88604_64"
            ],
            "features": torch.Tensor  # shape: [num_images_for_patient, 1024]
        }
    },
    "metadata": {
        "input_json": "data/OCT_eid.json",
        "checkpoint_path": "/data/home/wanglidi/model/RETFound_oct_weights.pth",
        "image_size": 224,
        "feature_layout": "features_by_eid[eid]['features'][i] matches features_by_eid[eid]['image_names'][i]"
    }
}
```

For each patient:

- `image_names` contains the image basename only, without path and extension.
- `features` is a tensor with one row per image.
- `features[i]` corresponds to `image_names[i]`.

This layout keeps feature tensors efficient while preserving exactly which OCT image produced each feature.

## 4. Load Encoded Features

Basic loading:

```python
import torch

encoded = torch.load("data/OCT_features.pt", map_location="cpu")
features_by_eid = encoded["features_by_eid"]

eid = "1791781"
image_names = features_by_eid[eid]["image_names"]
features = features_by_eid[eid]["features"]

print(image_names)
print(features.shape)  # [num_images_for_patient, 1024]
```

If a downstream model needs one feature vector per patient, a simple default is to average that patient's OCT image features:

```python
patient_feature = features.mean(dim=0)  # shape: [1024]
```

If the downstream model can use multiple OCT views/images, keep the full tensor:

```python
patient_image_features = features  # shape: [num_images_for_patient, 1024]
```

When joining with labels, use the eid string as the key:

```python
label_by_eid = {
    "1791781": 1,
    "1472882": 0,
}

rows = []
for eid, item in features_by_eid.items():
    if eid not in label_by_eid:
        continue
    patient_feature = item["features"].mean(dim=0)
    rows.append((eid, patient_feature, label_by_eid[eid]))
```

## Notes

- Run the extraction and encoding commands on the server where the OCT files and RETFound checkpoint exist.
- `encode.py` loads the model once and processes images in batches, so it is much faster than encoding one image at a time.
- If CUDA runs out of memory, reduce `--batch_size`.
- If some patients have no encoded images, check whether their zip files were missing, the selected middle image was absent, or the copied image paths in `data/OCT_eid.json` are no longer valid.
