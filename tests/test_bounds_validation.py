import pytest
from lss.models.lift_splat import LiftSplat
def test_lift_splat_rejects_invalid_bounds():
    with pytest.raises(ValueError): LiftSplat((1,1,1))
