from dataclasses import dataclass


@dataclass
class OfficialConfig:
    image_size: tuple = (256, 704)
    depth_bins: int = 80
    num_cameras: int = 6
    batch_size: int = 8
    epochs: int = 24
