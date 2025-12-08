"""
Utility functions for loading, converting, and manipulating images.

This module provides functions for:
- Converting between different image formats and representations
- Resizing and cropping images to specific resolutions
- Loading and normalizing images for model input
- Handling various image file formats including HEIF/HEIC when available
"""

import os

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from any4d.utils.moge_inference import run_moge_inference, load_moge_model

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

from any4d.utils.cropping import crop_resize_if_necessary
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

# Fixed resolution mappings with precomputed aspect ratios as keys
RESOLUTION_MAPPINGS = {
    518: {
        1.000: (518, 518),  # 1:1
        1.321: (518, 392),  # 4:3
        1.542: (518, 336),  # 3:2
        1.762: (518, 294),  # 16:9
        2.056: (518, 252),  # 2:1
        3.083: (518, 168),  # 3.2:1
        0.757: (392, 518),  # 3:4
        0.649: (336, 518),  # 2:3
        0.567: (294, 518),  # 9:16
        0.486: (252, 518),  # 1:2
    },
    512: {
        1.000: (512, 512),  # 1:1
        1.333: (512, 384),  # 4:3
        1.524: (512, 336),  # 3:2
        1.778: (512, 288),  # 16:9
        2.000: (512, 256),  # 2:1
        3.200: (512, 160),  # 3.2:1
        0.750: (384, 512),  # 3:4
        0.656: (336, 512),  # 2:3
        0.562: (288, 512),  # 9:16
        0.500: (256, 512),  # 1:2
    },
}

# Precomputed sorted aspect ratio keys for efficient lookup
ASPECT_RATIO_KEYS = {
    518: sorted(RESOLUTION_MAPPINGS[518].keys()),
    512: sorted(RESOLUTION_MAPPINGS[512].keys()),
}

def img_to_arr(img):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def find_closest_aspect_ratio(aspect_ratio, resolution_set):
    """
    Find the closest aspect ratio from the resolution mappings using efficient key lookup.

    Args:
        aspect_ratio (float): Target aspect ratio
        resolution_set (int): Resolution set to use (518 or 512)

    Returns:
        tuple: (target_width, target_height) from the resolution mapping
    """
    aspect_keys = ASPECT_RATIO_KEYS[resolution_set]

    # Find the closest aspect ratio key using binary search approach
    closest_key = min(aspect_keys, key=lambda x: abs(x - aspect_ratio))

    return RESOLUTION_MAPPINGS[resolution_set][closest_key]


def rgb(ftensor, norm_type, true_shape=None):
    """
    Convert normalized image tensor to RGB image for visualization.

    Args:
        ftensor (torch.Tensor or numpy.ndarray or list): Image tensor or list of image tensors
        norm_type (str): Normalization type, see UniCeption IMAGE_NORMALIZATION_DICT keys or use "identity"
        true_shape (tuple, optional): If provided, the image will be cropped to this shape (H, W)

    Returns:
        numpy.ndarray: RGB image with values in range [0, 1]
    """
    if isinstance(ftensor, list):
        return [rgb(x, norm_type, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        if norm_type in IMAGE_NORMALIZATION_DICT.keys():
            img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
            mean = img_norm.mean.numpy()
            std = img_norm.std.numpy()
        elif norm_type == "identity":
            mean = 0.0
            std = 1.0
        else:
            raise ValueError(
                f"Unknown image normalization type: {norm_type}. Available types: identity or {IMAGE_NORMALIZATION_DICT.keys()}"
            )
        img = ftensor * std + mean
    return img.clip(min=0, max=1)


def load_images(
    folder_or_list,
    resize_mode="fixed_mapping",
    size=None,
    norm_type="dinov2",
    patch_size=14,
    verbose=False,
    bayer_format=False,
    resolution_set=518,
    stride=1,
    compute_moge_mask=False,
    moge_model=None,
    binary_mask_path=None
):
    """
    Open and convert all images in a list or folder to proper input format for model

    Args:
        folder_or_list (str or list): Path to folder or list of image paths.
        resize_mode (str): Resize mode - "fixed_mapping", "longest_side", "square", or "fixed_size". Defaults to "fixed_mapping".
        size (int or tuple, optional): Required for "longest_side", "square", and "fixed_size" modes.
                                      - For "longest_side" and "square": int value for resize dimension
                                      - For "fixed_size": tuple of (width, height)
        norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to "dinov2".
        patch_size (int, optional): Patch size for image processing. Defaults to 14.
        verbose (bool, optional): If True, print progress messages. Defaults to False.
        bayer_format (bool, optional): If True, read images in Bayer format. Defaults to False.
        resolution_set (int, optional): Resolution set to use for "fixed_mapping" mode (518 or 512). Defaults to 518.
        stride (int, optional): Load every nth image from the input. stride=1 loads all images, stride=2 loads every 2nd image, etc. Defaults to 1.

    Returns:
        list: List of dictionaries containing image data and metadata
    """
    # Validate resize_mode and size parameter requirements
    valid_resize_modes = ["fixed_mapping", "longest_side", "square", "fixed_size"]
    if resize_mode not in valid_resize_modes:
        raise ValueError(
            f"Resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
        )

    if resize_mode in ["longest_side", "square", "fixed_size"] and size is None:
        raise ValueError(f"Size parameter is required for resize_mode='{resize_mode}'")

    # Validate size type based on resize mode
    if resize_mode in ["longest_side", "square"]:
        if not isinstance(size, int):
            raise ValueError(
                f"Size must be an int for resize_mode='{resize_mode}', got {type(size)}"
            )
    elif resize_mode == "fixed_size":
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(
                f"Size must be a tuple/list of (width, height) for resize_mode='fixed_size', got {size}"
            )
        if not all(isinstance(x, int) for x in size):
            raise ValueError(
                f"Size values must be integers for resize_mode='fixed_size', got {size}"
            )

    # Get list of image paths
    if isinstance(folder_or_list, str):
        # If folder_or_list is a string, assume it's a path to a folder
        if verbose:
            print(f"Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
    elif isinstance(folder_or_list, list):
        # If folder_or_list is a list, assume it's a list of image paths
        if verbose:
            print(f"Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list
    else:
        # If folder_or_list is neither a string nor a list, raise an error
        raise ValueError(f"Bad {folder_or_list=} ({type(folder_or_list)})")

    # Define supported image extensions
    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    # First pass: Load all images and collect aspect ratios
    loaded_images = []
    aspect_ratios = []
    for i, path in enumerate(folder_content):
        # Skip images based on stride
        if i % stride != 0:
            continue

        # Check if the file has a supported image extension
        if not path.lower().endswith(supported_images_extensions):
            continue

        try:
            if bayer_format:
                # If bayer_format is True, read the image in Bayer format
                color_bayer = cv2.imread(os.path.join(root, path), cv2.IMREAD_UNCHANGED)
                color = cv2.cvtColor(color_bayer, cv2.COLOR_BAYER_RG2BGR)
                img = PIL.Image.fromarray(color)
                img = exif_transpose(img).convert("RGB")
            else:
                # Otherwise, read the image normally
                img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert(
                    "RGB"
                )

            W1, H1 = img.size
            aspect_ratios.append(W1 / H1)
            loaded_images.append((path, img, W1, H1))

        except Exception as e:
            if verbose:
                print(f"Warning: Could not load {path}: {e}")
            continue

    # Check if any images were loaded
    if not loaded_images:
        raise ValueError("No valid images found")

    # Calculate average aspect ratio and determine target size
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    if verbose:
        print(
            f"Calculated average aspect ratio: {average_aspect_ratio:.3f} from {len(aspect_ratios)} images"
        )

    # Determine target size for all images based on resize mode
    if resize_mode == "fixed_mapping":
        # Resolution mappings are already compatible with their respective patch sizes
        # 518 mappings are divisible by 14, 512 mappings are divisible by 16
        target_width, target_height = find_closest_aspect_ratio(
            average_aspect_ratio, resolution_set
        )
        target_size = (target_width, target_height)
    elif resize_mode == "square":
        target_size = (
            round((size // patch_size)) * patch_size,
            round((size // patch_size)) * patch_size,
        )
    elif resize_mode == "longest_side":
        # Use average aspect ratio to determine size for all images
        # Longest side should be the input size
        if average_aspect_ratio >= 1:  # Landscape or square
            # Width is the longest side
            target_size = (
                size,
                round((size // patch_size) / average_aspect_ratio) * patch_size,
            )
        else:  # Portrait
            # Height is the longest side
            target_size = (
                round((size // patch_size) * average_aspect_ratio) * patch_size,
                size,
            )
    elif resize_mode == "fixed_size":
        # Use exact size provided, aligned to patch_size
        target_size = (
            (size[0] // patch_size) * patch_size,
            (size[1] // patch_size) * patch_size,
        )

    if verbose:
        print(
            f"Using target resolution {target_size[0]}x{target_size[1]} (W x H) for all images"
        )

    # Get the image normalization function based on the norm_type
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    # Second pass: Resize all images to the same target size
    imgs = []
    for path, img, W1, H1 in loaded_images:

        # Load binary mask if path is provided
        if binary_mask_path is not None:
            mask_img = cv2.imread(binary_mask_path)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            binary_mask = (mask_img > 0).astype(np.float32)

            # # Dilate binary_mask further
            kernel_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

        else:
            binary_mask = np.ones((H1, W1), dtype=np.float32)  # Default mask, all pixels are valid

        # check if we need to compute moge mask
        if compute_moge_mask:
            moge_img = np.array(PIL.Image.open(os.path.join(root, path)))  # (H, W, 3)
            transform = tvf.Compose([tvf.ToTensor()])
            input_moge_img = transform(moge_img).unsqueeze(0)  # (1, 3, H, W)
            moge_output = run_moge_inference(moge_model, input_moge_img, device="cuda")
            non_ambiguous_mask = moge_output["mask"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)
            additional_quantities = [non_ambiguous_mask, binary_mask]
            # img, additional_quantities = crop_resize_if_necessary(img, resolution=size, additional_quantities=additional_quantities)
            img, additional_quantities = crop_resize_if_necessary(img, resolution=target_size, additional_quantities=additional_quantities)
            non_ambiguous_mask = torch.tensor(additional_quantities[0]).bool()
            binary_mask = torch.tensor(additional_quantities[1]).bool()
        else:
            additional_quantities = None
            # img = crop_resize_if_necessary(img, resolution=size)[0]
            img = crop_resize_if_necessary(img, resolution=target_size)[0]
            non_ambiguous_mask = torch.tensor(np.ones_like(img)).bool()  # Default mask, all pixels are valid
            binary_mask = torch.tensor(np.ones_like(img)).bool()

        # Normalize image and add it to the list
        W2, H2 = img.size
        if verbose:
            print(f" - Adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")

        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                data_norm_type=[norm_type],
                non_ambiguous_mask=non_ambiguous_mask,
                binary_mask=binary_mask
            )
        )

    assert imgs, "No images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")

    return imgs
