from lss.models import parameter_count, trainable_parameter_count
def test_diagnostics_are_public(): assert callable(parameter_count) and callable(trainable_parameter_count)
