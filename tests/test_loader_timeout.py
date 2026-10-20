from lss.data.loader import build_loader
class Empty:
    def __len__(self): return 0
def test_loader_accepts_timeout(): assert build_loader(Empty(),timeout=3).timeout == 3
