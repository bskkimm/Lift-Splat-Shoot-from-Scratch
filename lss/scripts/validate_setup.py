from pathlib import Path


def validate_dataroot(dataroot):
    path = Path(dataroot).expanduser()
    if not path.exists(): raise FileNotFoundError(path)
    return path
