from lss.data.loader import build_loader
class Empty:
    def __len__(self): return 0
def test_persistent_workers_requires_workers(): assert not build_loader(Empty(),persistent_workers=True).persistent_workers
