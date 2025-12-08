"""
Util functions to run inference with MoGe
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress XFormers warnings

import numpy as np
import rerun as rr
import torch
import torchvision
import torchvision.transforms as tvf
from PIL import Image

from any4d.utils.viz import log_data_to_rerun, script_add_rerun_args


def load_moge_model(
    model_code_path: str = "any4d/external/MoGe",
    ckpt_path: str = "Ruicheng/moge-vitl",
    device="cuda",
):
    """
    Load the MoGe (ViT-L) model from huggingface hub (or load from local).
    """
    if not Path(model_code_path).exists():
        raise FileNotFoundError(f"MoGe code not found at {model_code_path}")
    import sys

    # Add the MoGe code to the system path
    sys.path.append(str(model_code_path))

    # Init the MoGe model
    from any4d.external.MoGe.moge.model.v1 import MoGeModel

    model = MoGeModel.from_pretrained(ckpt_path).to(device).eval()

    return model


@torch.no_grad()
def run_moge_inference(model: torch.nn.Module, image: torch.tensor, device="cuda"):
    """
    Run MoGe inference on a batch of images or single image.
    Output is a dictionary with the following keys:
    - points: (B, H, W, 3) # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
    - depth: (B, H, W) # scale-invariant depth map
    - mask: (B, H, W) # a binary mask for valid pixels
    - intrinsics: (B, 3, 3) # normalized camera intrinsics

    Args:
        model: MoGe model
        image: (B, 3, H, W) or (3, H, W) # RGB image in range [0, 1]
    """
    image = image.to(device)
    return model.infer(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--image_path", default='/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/benchmarking/monst3r/demo_data/lady-running/00000.jpg', type=str)
    parser.add_argument("--viz", action="store_true")
    script_add_rerun_args(parser)  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    # Setup Rerun if needed
    if args.viz:
        rr.script_setup(args, f"MoGe_Pred_Viz")
        rr.set_time_seconds("stable_time", 0)
        rr.log("moge", rr.ViewCoordinates.RDF, static=True)

    # Load the input data
    img = np.array(Image.open(args.image_path))  # (H, W, 3)
    transform = tvf.Compose([tvf.ToTensor()])
    input_img = transform(img).unsqueeze(0)  # (B, 3, H, W)

    # Load the model
    model = load_moge_model()

    # Run the model inference
    output = run_moge_inference(model, input_img)

    # Get the different outputs
    pts3d = output["points"].cpu().squeeze(0).numpy()  # (H, W, 3)
    depth = output["depth"].cpu().squeeze(0).numpy()  # (H, W)
    mask = output["mask"].cpu().squeeze(0).numpy()  # (H, W)
    intrinsics = output["intrinsics"].cpu().squeeze(0).numpy()  # (3, 3), normalized
    intrinsics[0, :] = intrinsics[0, :] * depth.shape[1]
    intrinsics[1, :] = intrinsics[1, :] * depth.shape[0]

    # Log prediction to Rerun
    if args.viz:
        base_name = "moge"
        log_data_to_rerun(
            image=img, depthmap=depth, pose=np.eye(4), intrinsics=intrinsics, base_name=base_name, mask=np.float32(mask)
        )
        # Log the predicted 3D points
        filtered_pts = pts3d[mask]
        filtered_pts_col = img[mask]
        pts_name = f"{base_name}/points"
        rr.log(
            pts_name,
            rr.Points3D(
                positions=filtered_pts.reshape(-1, 3),
                colors=filtered_pts_col.reshape(-1, 3),
            ),
        )
