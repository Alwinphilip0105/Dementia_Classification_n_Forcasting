"""CLI to run explainability analysis on trained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import densenet121

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.spectrograms import (
    MelSpecConfig,
    load_mono_resampled,
    log_mel_spectrogram,
)
from dementia_project.viz.explainability import (
    explain_densenet_with_integrated_gradients,
    visualize_attributions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["densenet"], default="densenet")
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--pytorch_checkpoint", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("runs/explainability"))
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--target_class", type=int, default=1, help="0 or 1")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    df = metadata_df.merge(
        splits_df[["audio_path", "split"]], on="audio_path", how="inner"
    )

    # Filter by target class and split
    test_df = df[(df["split"] == "test") & (df["label"] == args.target_class)]
    if len(test_df) == 0:
        print(f"No test samples found for class {args.target_class}")
        return

    # Sample a few examples
    sample_df = test_df.sample(
        n=min(args.num_samples, len(test_df)), random_state=1337
    ).reset_index(drop=True)

    print(f"Analyzing {len(sample_df)} samples for class {args.target_class}")

    # Load or create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    if args.pytorch_checkpoint is not None and args.pytorch_checkpoint.exists():
        checkpoint = torch.load(args.pytorch_checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)

    cfg = MelSpecConfig(max_audio_sec=10.0)

    results: list[dict] = []

    for idx, row in sample_df.iterrows():
        print(f"Processing sample {idx+1}/{len(sample_df)}: {row['audio_path']}")

        # Load and process audio
        wav = load_mono_resampled(str(row["audio_path"]), cfg.sample_rate_hz)
        spec = log_mel_spectrogram(wav, cfg)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        x = spec.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1, 3, H, W]

        # Get prediction
        with torch.no_grad():
            x_tensor = x.to(device)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()

        # Compute attributions
        attributions, metadata = explain_densenet_with_integrated_gradients(
            model=model,
            input_tensor=x_tensor,
            target_class=args.target_class,
            device=device,
        )

        # Save visualization
        sample_id = Path(row["audio_path"]).stem
        viz_path = (
            args.out_dir / f"attribution_{sample_id}_class{args.target_class}.png"
        )
        visualize_attributions(
            input_tensor=x_tensor,
            attributions=attributions,
            output_path=viz_path,
            title=f"Integrated Gradients - {sample_id} (Class {args.target_class})",
        )

        results.append(
            {
                "audio_path": str(row["audio_path"]),
                "true_label": int(row["label"]),
                "predicted_class": int(pred_class),
                "probabilities": probs[0].cpu().numpy().tolist(),
                "attribution_metadata": metadata,
                "visualization_path": str(viz_path),
            }
        )

    # Save results
    results_path = args.out_dir / "explainability_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved explainability results to: {results_path}")
    print(f"Generated {len(results)} attribution visualizations")


if __name__ == "__main__":
    main()
