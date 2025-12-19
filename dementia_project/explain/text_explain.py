"""Explainability for text baseline using Captum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from transformers import AutoModel, AutoTokenizer

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.text_features import build_text_dataframe


class TextClassifier(nn.Module):
    """Transformer-based text classifier (must match training architecture)."""

    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def load_model(model_dir: Path, model_name: str, device):
    """Load trained model from checkpoint."""
    model = TextClassifier(model_name, num_classes=2).to(device)
    checkpoint_path = model_dir / "model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Re-run training and save model checkpoint."
        )

    # Text Model Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # extract state_dict, all learnable parameters (weights and biases) per layer
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def explain_text_sample(
    model, tokenizer, text: str, true_label: int, device, max_length: int = 512
):
    """Generate attribution scores for a single text sample."""

    # Tokenizing the input string
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # get predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        pred_label = int(pred_idx[0].item())
        confidence = confidence[0].item()

    # IG attribution
    def forward_func(input_ids, attention_mask):
        return model(input_ids, attention_mask)

    # This is a wrapper that returns logits for target class, with an attention mask
    def predict_fn(input_ids):
        attn = (input_ids != tokenizer.pad_token_id).long()
        return forward_func(input_ids, attn)

    lig = LayerIntegratedGradients(predict_fn, model.encoder.embeddings.word_embeddings)

    # Compute attributions for predicted class
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=input_ids * 0,  # baseline = all zeros (pad tokens)
        target=pred_label,
        return_convergence_delta=True,
    )

    # Sum attributions across embedding dimension
    attributions = attributions.sum(dim=-1).squeeze(0)  # [seq_len]
    attributions = attributions.cpu().detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())

    # Filter out padding tokens
    valid_indices = (input_ids.squeeze(0) != tokenizer.pad_token_id).cpu().numpy()
    tokens = [t for t, v in zip(tokens, valid_indices) if v]
    attributions = attributions[valid_indices]

    return {
        "text": text,
        "tokens": tokens,
        "attributions": attributions,
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": confidence,
        "delta": delta.item(),
    }


def visualize_attributions(result: dict, out_path: Path):
    """Create HTML visualization of word attributions."""

    tokens = result["tokens"]
    attributions = result["attributions"]

    # Normalize attributions for visualization
    abs_max = np.abs(attributions).max()
    if abs_max > 0:
        normalized_attrs = attributions / abs_max
    else:
        normalized_attrs = attributions

    html_parts = [
        "<html><head><style>",
        ".word { display: inline-block; margin: 2px; padding: 3px; border-radius: 3px; }",
        ".positive { background-color: rgba(0, 255, 0, VAR); }",
        ".negative { background-color: rgba(255, 0, 0, VAR); }",
        "</style></head><body>",
        f"<h3>True Label: {'Dementia' if result['true_label'] == 1 else 'Control'}</h3>",
        f"<h3>Predicted: {'Dementia' if result['pred_label'] == 1 else 'Control'} "
        f"(confidence: {result['confidence']:.3f})</h3>",
        "<p>",
    ]

    for token, attr in zip(tokens, normalized_attrs):
        if attr > 0:
            opacity = abs(attr)
            html_parts.append(
                f'<span class="word positive" style="background-color: rgba(0, 255, 0, {opacity});">{token}</span>'
            )
        elif attr < 0:
            opacity = abs(attr)
            html_parts.append(
                f'<span class="word negative" style="background-color: rgba(255, 0, 0, {opacity});">{token}</span>'
            )
        else:
            html_parts.append(f'<span class="word">{token}</span>')

    html_parts.append("</p>")
    html_parts.append(
        "<p><small>Green = supports dementia prediction, Red = supports control prediction</small></p>"
    )
    html_parts.append("</body></html>")

    out_path.write_text("\n".join(html_parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--asr_manifest_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_model(args.model_dir, args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading data...")
    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    asr_manifest_df = pd.read_csv(args.asr_manifest_csv)

    text_df = build_text_dataframe(metadata_df, asr_manifest_df)
    df = text_df.merge(splits_df[["audio_path", "split"]], on="audio_path", how="inner")

    # Use validation set for explanations (better balance than test)
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)
    examples = valid_df.sample(n=min(args.num_examples, len(valid_df)), random_state=42)

    ## Make Visualizations
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for idx, row in examples.iterrows():
        print(f"\nExplaining example {idx + 1}/{len(examples)}...")
        result = explain_text_sample(
            model, tokenizer, row["text"], row["label"], device, args.max_length
        )
        results.append(result)

        # save each generation to load in final report
        vis_path = args.out_dir / f"example_{idx}_attribution.html"
        visualize_attributions(result, vis_path)
        print(f"  Saved: {vis_path}")

    summary = {
        "num_examples": len(results),
        "examples": [
            {
                "text_preview": r["text"][:200] + "...",
                "true_label": r["true_label"],
                "pred_label": r["pred_label"],
                "confidence": r["confidence"],
                "convergence_delta": r["delta"],
            }
            for r in results
        ],
    }

    summary_path = args.out_dir / "explanations_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
