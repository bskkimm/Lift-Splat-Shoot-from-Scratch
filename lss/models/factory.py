from .lss import LSS


def build_model(**kwargs):
    if kwargs.get("depth_bins", 1) <= 0: raise ValueError("depth_bins must be positive")
    return LSS(**kwargs)
