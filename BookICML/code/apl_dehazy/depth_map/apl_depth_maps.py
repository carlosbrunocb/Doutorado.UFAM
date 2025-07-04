import torch
import numpy as np


def generate_depth_maps(batch_rgb_np: np.ndarray, device, processor, model,
                        normalize_output: bool = True) -> np.ndarray:
    """
    Generates depth maps using Depth Anything model (by PyTorch) from a batch of normalized rgb images .

    Args:
        :param batch_rgb_np: Batch of images with shape (B, H, W, 3) and normalized values in [0, 1].
        :param device: Device (CPU or CUDA).
        :param processor: Image processor loaded.
        :param model: Loaded depth estimation model.
        :param normalize_output: If True, normalize depth map model to [0,1].

    Returns:
        np.ndarray: depth maps in format (B, H, W), float32 or uint8.
    """
    assert batch_rgb_np.ndim == 4 and batch_rgb_np.shape[-1] == 3, "Expected pattern (B, H, W, 3)"

    depth_maps = []

    # Converter cada imagem individualmente
    for i, n_img in enumerate(batch_rgb_np):
        print(f"Processing the depth map of image {i}")
        img = (n_img * 255).astype(np.uint8)  # [0,1] -> [0,255]

        # Applies the model processor to preprocess
        inputs = processor(images=img, return_tensors="pt").to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.predicted_depth.squeeze(0).cpu().numpy()

            # Normalize (opcional)
            if normalize_output:
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            depth_maps.append(pred)

    return np.stack(depth_maps)


# Generating the binary masks
def generate_depth_masks_from_intervals(depth_maps: np.ndarray, num_intervals: int) -> np.ndarray:
    """
    Generate binary masks from normalized depth maps.

    Args:
        depth_maps (np.ndarray): Normalized depth maps in the format
            - (H, W)
            - (H, W, 1)
            - (B, H, W)
            - (B, H, W, 1)
        num_intervals (int): Number of bands that divide the interval between [0, 1].

    Returns:
        np.ndarray: Binary masks in the format (B, num_intervals, H, W) or (num_intervals, H, W).
    """
    # Identify and adjust the input format
    squeeze_output = False

    if depth_maps.ndim == 2:
        # (H, W) → add batch
        depth_maps = depth_maps[None, ...]
        squeeze_output = True

    elif depth_maps.ndim == 3:
        if depth_maps.shape[-1] == 1:
            # (H, W, 1) → remove channel, add batch
            depth_maps = depth_maps[..., 0][None, ...]
            squeeze_output = True
        else:
            # (B, H, W)
            pass

    elif depth_maps.ndim == 4 and depth_maps.shape[-1] == 1:
        # (B, H, W, 1) → remove channel
        depth_maps = depth_maps[..., 0]

    else:
        raise ValueError("Invalid input format. Expected (H, W), (H, W, 1), (B, H, W), or (B, H, W, 1).")

    B, H, W = depth_maps.shape
    masks = np.zeros((B, num_intervals, H, W), dtype=np.float32)

    interval_edges = np.linspace(0, 1, num_intervals + 1)

    for i in range(num_intervals):
        lower = interval_edges[i]
        upper = interval_edges[i + 1]
        # Create mask for range [lower, upper)
        masks[:, i, :, :] = ((depth_maps >= lower) & (depth_maps < upper)).astype(np.float32)

    if squeeze_output:
        return masks[0]  # return (num_intervals, H, W)

    return masks  # return (B, num_intervals, H, W)
