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
    v0, e0, x0 = get_inputs()
    v1, e1, x1 = Block()(v0, e0, x0)
    assert_dimension_reasonable(v0, e0, x0, v1, e1, x1)

@pytest.mark.parametrize("Block", list(aperol.blocks.all_blocks.values()))
def test_equivariance(Block, *args):
    block = Block()
    config = block.sample()
    v0, e0, x0 = get_inputs()
    v1, e1, x1 = block(v0, e0, x0, config=config)
    translation, rotation, reflection = get_translation_rotation_reflection()

    # test translation
    x0_translation = translation(x0)
    v1_translation, e1_translation, x1_translation = block(
        v0, e0, x0_translation, config=config
    )

    if Block.__name__ != "DotProductReduce":
        assert torch.allclose(v1_translation, v1, atol=1e-3, rtol=1e-3)
    if Block.__name__ != "SpatialAttention":
        assert torch.allclose(e1_translation, e1, atol=1e-3, rtol=1e-3)
    if Block.__name__ != "GeometryReduce":
        assert torch.allclose(
            x1_translation[..., 0], translation(x1)[..., 0],
            atol=1e-3, rtol=1e-3
        )

    x0_rotation = rotation(x0)
    v1_rotation, e1_rotation, x1_rotation = block(
        v0, e0, x0_rotation, config=config,
    )
    assert torch.allclose(v1_rotation, v1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(e1_rotation, e1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(x1_rotation, rotation(x1), atol=1e-3, rtol=1e-3)

    x0_reflection = reflection(x0)
    v1_reflection, e1_reflection, x1_reflection = block(
        v0, e0, x0_reflection, config=config,
    )
    assert torch.allclose(v1_reflection, v1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(e1_reflection, e1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(x1_reflection, reflection(x1), atol=1e-3, rtol=1e-3)
