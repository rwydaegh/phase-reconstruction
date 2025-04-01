# tests/perturbations/test_momentum.py

import os

# Make sure src is importable
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.perturbations.momentum import apply_momentum_perturbation


def test_momentum_perturbation_shape_and_type():
    """Test if momentum perturbation returns correct shapes and types."""
    field_values = np.ones((10, 10), dtype=complex)
    iteration = 1
    intensity = 0.1
    current_error = 0.5
    previous_momentum = None
    perturbed_values, new_momentum = apply_momentum_perturbation(
        field_values, current_error, previous_momentum, iteration, intensity
    )

    assert perturbed_values.shape == field_values.shape
    assert new_momentum.shape == field_values.shape
    assert perturbed_values.dtype == np.complex128 or perturbed_values.dtype == np.complex64
    assert new_momentum.dtype == np.complex128 or new_momentum.dtype == np.complex64

def test_momentum_perturbation_changes_values():
    """Test if momentum perturbation changes values (non-zero intensity)."""
    field_values = np.ones((10, 10), dtype=complex) * (1+1j)
    iteration = 1
    intensity = 0.1
    current_error = 0.5
    previous_momentum = None
    perturbed_values, _ = apply_momentum_perturbation(
        field_values, current_error, previous_momentum, iteration, intensity
    )

    assert not np.allclose(perturbed_values, field_values)

def test_momentum_perturbation_zero_intensity():
    """Test if zero intensity results in no change."""
    field_values = np.ones((10, 10), dtype=complex) * (1+1j)
    iteration = 1
    intensity = 0.0
    current_error = 0.5
    previous_momentum = None
    perturbed_values, new_momentum = apply_momentum_perturbation(
        field_values, current_error, previous_momentum, iteration, intensity
    )

    # With zero intensity, the perturbation should be zero
    assert np.allclose(perturbed_values, field_values)
    # The new momentum should also be zero in this case
    assert np.allclose(new_momentum, np.zeros_like(field_values))


def test_momentum_perturbation_with_previous_momentum():
    """Test if providing previous momentum influences the result."""
    field_values = np.ones((5, 5), dtype=complex) * (1+1j)
    iteration = 1
    intensity = 0.1
    current_error = 0.5
    previous_momentum = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)

    perturbed_values_no_mom, new_momentum_no_mom = apply_momentum_perturbation(
        field_values, current_error, None, iteration, intensity, momentum_factor=0.8
    )
    perturbed_values_with_mom, new_momentum_with_mom = apply_momentum_perturbation(
        field_values, current_error, previous_momentum, iteration, intensity, momentum_factor=0.8
    )

    # The results should differ when previous momentum is provided vs. when it's None
    assert not np.allclose(perturbed_values_no_mom, perturbed_values_with_mom)
    assert not np.allclose(new_momentum_no_mom, new_momentum_with_mom)
    # The new momentum with previous should not be identical to the random part alone
    assert not np.allclose(new_momentum_with_mom, new_momentum_no_mom)


def test_momentum_perturbation_zero_field():
    """Test handling of zero input field."""
    field_values = np.zeros((10, 10), dtype=complex)
    iteration = 1
    intensity = 0.1
    current_error = 0.5
    previous_momentum = None
    perturbed_values, new_momentum = apply_momentum_perturbation(
        field_values, current_error, previous_momentum, iteration, intensity
    )

    # Perturbation should still be applied (random component)
    assert not np.allclose(perturbed_values, field_values)
