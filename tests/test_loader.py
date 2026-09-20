from lss.data.loader import build_loader
class Empty:
    def __len__(self): return 0
def test_loader_builder_returns_dataloader(): assert build_loader(Empty()).batch_size == 1
