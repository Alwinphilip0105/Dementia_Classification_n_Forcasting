python3 -m pip install --upgrade pip
python3 -m pip install poetry
poetry install

# Build metadata by joining WAV filenames to `DementiaNet - dementia.csv`.
poetry run python -m dementia_project.data.build_metadata --dementia_dir "dementia-20251217T041331Z-1-001" --control_dir "nodementia-20251217T041501Z-1-001" --dementia_csv "DementiaNet - dementia.csv" --out_dir "data/processed"  

# Build splits
poetry run python -m dementia_project.data.build_splits --metadata_csv "data/processed/metadata.csv" --out_dir "data/processed"

# Run ASR (generate transcripts)
poetry run python -m dementia_project.asr.run_asr --metadata_csv "data/processed/metadata.csv" --out_dir "data/processed/asr_whisper" --model_name "openai/whisper-tiny" --language en --task transcribe

# Train No ML Baselines
poetry run python -m dementia_project.train.train_nonml --metadata_csv "data/processed/metadata.csv" --splits_csv "data/processed/splits.csv" --out_dir "runs/nonml_baseline_scaled"

# Train Wav2Vec2 Baseline
poetry run python -m dementia_project.train.train_wav2vec2_nonml --metadata_csv "data/processed/metadata.csv" --splits_csv "data/processed/splits.csv" --out_dir "runs/wav2vec2_baseline_full_cuda" --max_audio_sec 10

# Train Densenet on Spectograms
poetry run python -m dementia_project.train.train_densenet_spec --metadata_csv "data/processed/metadata.csv" --splits_csv "data/processed/splits.csv" --out_dir "runs/densenet_spec_full_cuda" --epochs 5 --batch_size 16

# Train Text Only Baseline
poetry run python -m dementia_project.train.train_text_baseline --metadata_csv "data/processed/metadata.csv" --splits_csv "data/processed/splits.csv" --asr_manifest_csv "data/processed/asr_whisper/asr_manifest.csv" --out_dir "runs/text_baseline_roberta"

# Export Models with ONNX
poetry run python -m dementia_project.export.run_onnx_export --model_type densenet --out_dir artifacts --test

# Run Explainability and Generate Visualizations
poetry run python -m dementia_project.viz.run_explainability --model_type densenet --metadata_csv "data/processed/metadata.csv" --splits_csv "data/processed/splits.csv" --out_dir "runs/explainability" --num_samples 5

