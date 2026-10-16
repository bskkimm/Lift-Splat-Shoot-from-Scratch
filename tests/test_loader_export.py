from lss.data import build_loader
def test_loader_builder_is_public(): assert callable(build_loader)
