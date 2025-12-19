"""Model definitions (baselines + fusion)."""

from dementia_project.models.fusion_model import (
    CrossAttentionFusion,
    MultimodalFusionClassifier,
)

__all__ = ["CrossAttentionFusion", "MultimodalFusionClassifier"]
