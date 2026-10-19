from lss.data.loader import build_loader
class Empty:
    def __len__(self): return 0
def test_loader_accepts_drop_last(): assert build_loader(Empty(),drop_last=True).drop_last
