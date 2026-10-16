import pytest
from lss.geometry import make_frustum
def test_frustum_rejects_invalid_dimensions():
    with pytest.raises(ValueError): make_frustum(0, 2, [1])
