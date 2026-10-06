from lss.data.loader import build_loader
class Empty:
    def __len__(self): return 0
def test_loader_accepts_pin_memory(): assert build_loader(Empty(),pin_memory=True).pin_memory
