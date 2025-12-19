"""CLI to export best model to ONNX and run conformance test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dementia_project.export.onnx_export import export_densenet_to_onnx
from dementia_project.export.test_onnx import test_densenet_onnx_conformance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["densenet", "fusion"],
        default="densenet",
        help="Model type to export",
    )
    parser.add_argument(
        "--pytorch_checkpoint",
        type=Path,
        default=None,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test", action="store_true", help="Run conformance test")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "densenet":
        onnx_path = args.out_dir / "densenet_model.onnx"
        print(f"Exporting DenseNet to ONNX: {onnx_path}")
        export_densenet_to_onnx(
            model_path=args.pytorch_checkpoint,
            output_path=onnx_path,
            device="cpu",
        )

        if args.test:
            print("\nRunning conformance test...")
            results = test_densenet_onnx_conformance(
                onnx_path=onnx_path,
                pytorch_model_path=args.pytorch_checkpoint,
                num_test_samples=10,
                tolerance=1e-4,
            )
            results_path = args.out_dir / "onnx_conformance_test.json"
            # Convert numpy types to native Python types for JSON
            results_serializable = {
                k: (int(v) if isinstance(v, (bool, np.bool_)) else v)
                for k, v in results.items()
            }
            results_path.write_text(json.dumps(results_serializable, indent=2))
            print(f"Saved test results to: {results_path}")

    elif args.model_type == "fusion":
        from dementia_project.export.onnx_export import export_fusion_model_to_onnx
        from dementia_project.export.test_onnx import test_fusion_onnx_conformance

        onnx_path = args.out_dir / "fusion_model.onnx"
        print(f"Exporting Fusion model to ONNX: {onnx_path}")
        export_fusion_model_to_onnx(
            model_path=args.pytorch_checkpoint,
            output_path=onnx_path,
            device="cpu",
        )

        if args.test:
            print("\nRunning conformance test...")
            results = test_fusion_onnx_conformance(
                onnx_path=onnx_path,
                pytorch_model_path=args.pytorch_checkpoint,
                num_test_samples=10,
                tolerance=1e-4,
            )
            results_path = args.out_dir / "fusion_onnx_conformance_test.json"
            # Convert numpy types to native Python types for JSON
            results_serializable = {
                k: (int(v) if isinstance(v, (bool, np.bool_)) else v)
                for k, v in results.items()
            }
            results_path.write_text(json.dumps(results_serializable, indent=2))
            print(f"Saved test results to: {results_path}")

    print(f"\nONNX export complete. Model saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
