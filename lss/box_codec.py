import torch


def encode_yaw(yaw):
    return torch.stack((torch.sin(yaw), torch.cos(yaw)), -1)


def decode_yaw(encoded):
    return torch.atan2(encoded[..., 0], encoded[..., 1])
