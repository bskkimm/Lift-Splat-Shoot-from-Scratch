from lss import build_optimizer, build_scheduler
def test_optim_helpers_are_public(): assert callable(build_optimizer) and callable(build_scheduler)
