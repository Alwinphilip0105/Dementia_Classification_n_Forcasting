"""Tests for text classifier model."""

import torch

from dementia_project.train.train_text_baseline import TextClassifier


def test_text_classifier_forward():
    """Test model forward pass."""
    model = TextClassifier("distilbert-base-uncased", num_classes=2)
    batch_size = 2
    seq_length = 128

    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    logits = model(input_ids, attention_mask)
    assert logits.shape == (batch_size, 2)
