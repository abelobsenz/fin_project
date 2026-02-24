# Workflow Context Bundle

This folder is a curated snapshot of the files most relevant to the end-to-end `ivdyn` workflow, so you can share focused context with ChatGPT.

## Included workflow code

- `README.md`: pipeline overview, commands, expected artifacts.
- `pyproject.toml`: package + CLI entrypoint config.
- `src/ivdyn/utils/cli.py`: command surface (`pull-*`, `build-dataset`, `train`, `evaluate`, `backtest`, `ui`, `wf`).
- `src/ivdyn/data/schemas.py`: canonical options-chain schema and normalization.
- `src/ivdyn/data/massive.py`: data plugins and flatfile parsing.
- `src/ivdyn/data/loaders.py`: dataset builder and feature/target construction.
- `src/ivdyn/surface/build.py`: surface construction and interpolation/repair.
- `src/ivdyn/model/torch_system.py`: model definition and bundle IO.
- `src/ivdyn/training/pipeline.py`: staged model training.
- `src/ivdyn/eval/pipeline.py`: evaluation artifacts and metrics.

## Included data-format references

- `data_samples/day_aggs_2025-03-20.metadata.json`
- `data_samples/minute_aggs_2025-03-20.metadata.json`
- `data_samples/dataset_meta_day_aggs_train_2024.json`
- `data_samples/dataset_meta_minute_aggs_train_2024.json`
- `DATA_FORMAT_DAY_MINUTE.md` (written summary of day/minute formats)

