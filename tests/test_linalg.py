"""Tests for linear-algebra utilities."""

import pytest
import torch

from jmstate.utils._checks import check_matrix_dim
from jmstate.utils._linalg import flat_from_log_cholesky, log_cholesky_from_flat


def test_cholesky():
    matrix = torch.tensor([[1.0, 0.0], [0.2, 2.0]])
    flat = flat_from_log_cholesky(matrix, "full")
    torch.testing.assert_close(log_cholesky_from_flat(flat, 2, "full"), matrix)


def test_invalid_precision():
    with pytest.raises(ValueError, match="Precision type"):
        check_matrix_dim(torch.ones(1), 2, "dense")
