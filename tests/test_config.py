from lss.config import OfficialConfig
def test_official_config_defaults():
    config = OfficialConfig(); assert config.num_cameras == 6 and config.depth_bins == 80
