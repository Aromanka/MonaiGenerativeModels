# Synthetic EHR-to-OCT dataset factory

This folder implements three independent, restartable stages:

1. `generate_oct.py` reads `payload["trajectories"]`, validates each 120-D
   `ehr_latent`, runs `python -m oct_ehr_ldm sample`, and organizes generated
   PNGs plus `oct_manifest.csv`.
2. `encode_oct.py` invokes the existing `encoder/encode.py`, preserving its
   RETFound ViT-Large checkpoint and preprocessing exactly, then validates the
   expected 1024-D output.
3. `build_schema_v2.py` uses `xdiabetes2`'s `EncodedSample`, `EncoderOutput`,
   and `build_v2_payload` APIs to write the final training artifact.

`run_pipeline.sh` calls all stages. From the repository root:

```bash
bash factory/run_pipeline.sh \
  --ehr-pickle Dataset/generated/ukb_train_trajectories.pkl \
  --generator-checkpoint outputs/diffusion_full/best.pt \
  --autoencoder-checkpoint outputs/autoencoder/best.pt \
  --retfound-checkpoint /data/home/wanglidi/model/RETFound_oct_weights.pth \
  --schema-project-root ../../xdiabetes2 \
  --output-root Dataset/synthetic/ukbehr_ehr_oct \
  --samples-per-view 4 \
  --guidance-scale 4.0 \
  --inference-steps 50 \
  --seed 42
```

The defaults generate both UKB OCT views and map field `21017` to `left` and
`21018` to `right`. Override or extend this mapping with repeated arguments,
for example `--view-code 21017 --view-laterality 21017=left`.

The unified pipeline requires `--autoencoder-checkpoint` and passes that exact
path to OCT sampling. It does not use `paths.oct_autoencoder_checkpoint` from
the JSON config. The lower-level `python -m oct_ehr_ldm sample` command still
falls back to the config value when its optional `--autoencoder-checkpoint` is
omitted.

By default, source EIDs are preserved. To assign a disjoint sequential ID
range (for example for a validation dataset), use:

```bash
--patient-id-mode sequential --sequential-start 20000001
```

The source-to-output mapping is always retained in `patient_map.json` and in
each Schema V2 sample's metadata. `--offset`, `--limit`, and repeated
`--patient-id` arguments can restrict a run. Existing completed stages are
reused; pass `--force` to rerun them. To resume from a later stage, pass
`--start-stage encode` or `--start-stage package`.

The resulting layout is:

```text
OUTPUT_ROOT/
├── images/<patient_id>/*.png
├── oct_manifest.csv
├── oct_image_index.json
├── patient_map.json
├── OCT_features_synthetic.pt
├── dataset_manifest.json
└── ukb_synthetic_train.pt
```

Each `synthetic_NNNN` index is a separate Schema V2 visit. Its left/right OCT
features are retained as separate 1024-D OCT tokens, while the corresponding
120-D EHR latent is one EHR token. If a trajectory has no `cutoff_age_days`,
the factory records and uses `cutoff_age_years * 365.25`.

Individual stages can also be invoked with `python -m factory.generate_oct`,
`python -m factory.encode_oct`, and `python -m factory.build_schema_v2`; use
`--help` for their complete options.
