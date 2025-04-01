# tests/perturbations/test_basic.py

import os

# Make sure src is importable
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.perturbations.basic import apply_basic_perturbation


def test_apply_basic_perturbation_shape_and_type():
    """Test if the basic perturbation returns the correct shape and dtype."""
    field_values = np.ones((10, 10), dtype=complex)
    iteration = 1
    intensity = 0.1
    perturbed_values = apply_basic_perturbation(field_values, iteration, intensity)

    assert perturbed_values.shape == field_values.shape
    assert perturbed_values.dtype == np.complex128 or perturbed_values.dtype == np.complex64

def test_apply_basic_perturbation_changes_values():
    """Test if the perturbation actually changes the values (for non-zero intensity)."""
    field_values = np.ones((10, 10), dtype=complex) * (1+1j)
    iteration = 1
    intensity = 0.1
    perturbed_values = apply_basic_perturbation(field_values, iteration, intensity)

    # Check that the output is not identical to the input
    assert not np.allclose(perturbed_values, field_values)

def test_apply_basic_perturbation_zero_intensity():
    """Test if zero intensity results in no change."""
    field_values = np.ones((10, 10), dtype=complex) * (1+1j)
    iteration = 1
    intensity = 0.0
    perturbed_values = apply_basic_perturbation(field_values, iteration, intensity)

    # Check that the output is identical to the input
    assert np.allclose(perturbed_values, field_values)

def test_apply_basic_perturbation_zero_field():
    """Test handling of zero input field."""
    field_values = np.zeros((10, 10), dtype=complex)
    iteration = 1
    intensity = 0.1
    perturbed_values = apply_basic_perturbation(field_values, iteration, intensity)

    # Check that the output is still zero
    assert np.allclose(perturbed_values, field_values)
    assert np.allclose(perturbed_values, np.zeros_like(field_values))
