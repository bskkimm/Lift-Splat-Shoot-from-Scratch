from lss import focal_loss, lss_loss
def test_loss_functions_are_public(): assert callable(focal_loss) and callable(lss_loss)
