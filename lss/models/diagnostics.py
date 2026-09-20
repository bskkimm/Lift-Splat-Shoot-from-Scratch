def parameter_count(model, trainable_only=True):
    return sum(parameter.numel() for parameter in model.parameters() if not trainable_only or parameter.requires_grad)
