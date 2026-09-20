from lss.data.loader import build_loader
class Empty:
    def __len__(self): return 0
def test_loader_accepts_seed(): assert build_loader(Empty(),seed=4).worker_init_fn is not None
