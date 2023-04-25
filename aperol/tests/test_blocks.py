import pytest
import torch
import aperol
from utils import (
    get_inputs,
    assert_dimension_reasonable,
    get_translation_rotation_reflection,
)

@pytest.mark.parametrize("Block", list(aperol.blocks.all_blocks.values()))
def test_dimension_reasonable(Block, *args):
    v0, e0, x0, p0 = get_inputs()
    v1, e1, x1, p1 = Block()(v0, e0, x0, p0)
    assert_dimension_reasonable(v0, e0, x0, p0, v1, e1, x1, p1)

@pytest.mark.parametrize("Block", list(aperol.blocks.all_blocks.values()))
def test_equivariance(Block, *args):
    block = Block()
    config = block.sample()
    v0, e0, x0, p0 = get_inputs()
    v1, e1, x1, p1 = block(v0, e0, x0, p0, config=config)
    translation, rotation, reflection = get_translation_rotation_reflection()

    # test translation
    x0_translation = translation(x0)
    v1_translation, e1_translation, x1_translation, p1_translation = block(
        v0, e0, x0_translation, p0, config=config
    )

    assert torch.allclose(v1_translation, v1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(e1_translation, e1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(
        x1_translation, translation(x1),
        atol=1e-3, rtol=1e-3
    )

    x0_rotation = rotation(x0)
    p0_rotation = rotation(p0)
    v1_rotation, e1_rotation, x1_rotation, p0_rotation = block(
        v0, e0, x0_rotation, p0_rotation, config=config,
    )
    assert torch.allclose(v1_rotation, v1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(e1_rotation, e1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(x1_rotation, rotation(x1), atol=1e-3, rtol=1e-3)

    x0_reflection = reflection(x0)
    p0_reflection = reflection(p0)
    v1_reflection, e1_reflection, x1_reflection, p1_reflection = block(
        v0, e0, x0_reflection, p0_reflection, config=config,
    )
    assert torch.allclose(v1_reflection, v1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(e1_reflection, e1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(x1_reflection, reflection(x1), atol=1e-3, rtol=1e-3)
