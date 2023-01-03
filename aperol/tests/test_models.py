import pytest
import torch
import aperol
from utils import (
    get_model_inputs,
    assert_dimension_reasonable,
    get_translation_rotation_reflection,
)

@pytest.mark.parametrize("repeat", range(8))
def test_dimension_reasonable(repeat):
    v0, x0 = get_model_inputs()
    model = aperol.models.SuperModel(v0.shape[-1], 1)
    v1, x1 = model(v0, x0)
    assert v1.shape[0] == x1.shape[0]
