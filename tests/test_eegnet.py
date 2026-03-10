"""Tests for EEGNet architecture and forward pass."""

import numpy as np
import pytest
import torch

from src.models.eegnet import EEGNet, count_parameters


def test_eegnet_forward_shape():
    """Instantiate EEGNet and verify output shape for a random input."""
    n_channels = 64
    n_timepoints = 721  # matches -0.5 to 4.0 s at 160 Hz

    model = EEGNet(n_channels=n_channels, n_timepoints=n_timepoints)

    batch_size = 4
    x = torch.randn(batch_size, 1, n_channels, n_timepoints)
    out = model(x)

    assert out.shape == (batch_size, 2), f"Expected (4, 2), got {out.shape}"


def test_eegnet_parameter_count():
    """Verify EEGNet has a compact architecture."""
    model = EEGNet(n_channels=64, n_timepoints=721)
    n_params = count_parameters(model)
    # EEGNet should be compact — typically a few thousand params
    assert n_params > 0
    # Sanity: should be under 50K for this config
    assert n_params < 50_000, f"Too many parameters: {n_params}"


def test_eegnet_different_input_sizes():
    """EEGNet should adapt to different channel/timepoint configurations."""
    for n_ch, n_tp in [(32, 400), (64, 721), (8, 200)]:
        model = EEGNet(n_channels=n_ch, n_timepoints=n_tp)
        x = torch.randn(2, 1, n_ch, n_tp)
        out = model(x)
        assert out.shape == (2, 2)


def test_eegnet_output_logits():
    """Output should be raw logits (not probabilities) — no built-in softmax."""
    model = EEGNet(n_channels=64, n_timepoints=721)
    x = torch.randn(2, 1, 64, 721)
    out = model(x)
    # Logits can be negative or > 1
    # Just verify they're finite
    assert torch.isfinite(out).all()
