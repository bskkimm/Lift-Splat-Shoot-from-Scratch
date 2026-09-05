from lss.models.factory import build_model
from lss.models.lss import LSS
def test_factory_builds_lss(): assert isinstance(build_model(depth_bins=2), LSS)
