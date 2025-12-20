"""Model definitions (baselines + fusion)."""

from dementia_project.models.fusion_model import (
    CrossAttentionFusion,
    FusionClassifier,
)

# Backward compatibility alias
MultimodalFusionClassifier = FusionClassifier

__all__ = ["CrossAttentionFusion", "FusionClassifier", "MultimodalFusionClassifier"]
