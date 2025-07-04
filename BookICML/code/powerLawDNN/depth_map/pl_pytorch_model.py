import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor


def load_depth_map_anything(version='v1'):
    """
    Args:
        :param version: version of Depth Anything model

    Returns:
        :return device: Device (CPU or CUDA).
        :return processor: Image processor loaded.
        :return model: Loaded depth estimation model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match version:
        case 'v1':
            print('version 1.0 selected')
            processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf", use_fast=True)
            model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
            model.eval()
            return device, processor, model
        case 'v2':
            print('version 2.0 selected')
            processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf", use_fast=True)
            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to(device)
            model.eval()
            return device, processor, model
        case _:
            print("The passed parameter is invalid!")
            print("The model can not be built!")
