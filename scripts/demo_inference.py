# --------------------------------------------------------
# Sample multi-view inference script for Any4D
# --------------------------------------------------------
import os
import cv2
import hydra
import rerun as rr
import torch
from natsort import natsorted
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from glob import glob

from any4d.utils.inference import loss_of_one_batch_multi_view
from any4d.models import init_model
from any4d.utils.geometry import (
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
    normals_edge,
    depth_edge,
    points_to_normals,
)
from any4d.utils.image import load_images, rgb
from any4d.utils.misc import seed_everything
from any4d.utils.viz import script_add_rerun_args


# Loading Moge model
from any4d.utils.moge_inference import load_moge_model


def log_data_to_rerun(image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None):
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
            from_parent=False,
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]
    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
            # radii=0.02,
        ),
    )


def log_points_to_rerun(image, pts3d, pts_name, mask=None):
    # Log points in 3D
    if mask is None:
        filtered_pts = pts3d
        filtered_pts_col = image
    else:
        filtered_pts = pts3d[mask]
        filtered_pts_col = image[mask]
    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )


def init_hydra_config(config_path, overrides=None):
    "Initialize Hydra config"
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).split(".")[0]
    relative_path = os.path.relpath(config_dir, os.path.dirname(__file__))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=relative_path)
    if overrides is not None:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        cfg = hydra.compose(config_name=config_name)

    return cfg

def log_scene_flow_to_rerun(image, pts3d, scene_flow_vecs, base_name, mask=None, scene_bounds=None, colormap_name='rainbow'):
    """
    Log scene flow in 3D with color visualization based on flow magnitude and direction
    """
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.colors import hsv_to_rgb
    import torch

    # Filter points based on mask if provided
    if mask is None:
        filtered_pts = pts3d
        filtered_scene_flow_vecs = scene_flow_vecs
        filtered_pts_col = image
    else:
        filtered_pts = pts3d[mask]
        filtered_scene_flow_vecs = scene_flow_vecs[mask]
        filtered_pts_col = image[mask]
    
    # Check if we have any valid points
    if filtered_pts.numel() == 0:
        print(f"Warning: No valid points found for {base_name}. Skipping visualization.")
        return scene_bounds
    
    # Reshape tensors
    filtered_pts = filtered_pts.reshape(-1, 3)
    filtered_scene_flow_vecs = filtered_scene_flow_vecs.reshape(-1, 3)
    
    # Sample a subset of points to avoid overcrowding visualization
    max_arrows = 10000
    if filtered_pts.shape[0] > max_arrows:
        flow_magnitudes = torch.norm(filtered_scene_flow_vecs, dim=1)
        
        if flow_magnitudes.max() > 1e-6:
            probabilities = 0.2 + 0.8 * (flow_magnitudes / (flow_magnitudes.max() + 1e-6))
            probabilities_np = probabilities.cpu().numpy()
            probabilities_np = probabilities_np / probabilities_np.sum()
            
            indices_np = np.random.choice(
                filtered_pts.shape[0], 
                size=max_arrows, 
                replace=False, 
                p=probabilities_np
            )
            indices = torch.tensor(indices_np, device=filtered_pts.device)
        else:
            indices = torch.randperm(filtered_pts.shape[0], device=filtered_pts.device)[:max_arrows]
            
        sampled_pts = filtered_pts[indices]
        sampled_vectors = filtered_scene_flow_vecs[indices]
    else:
        sampled_pts = filtered_pts
        sampled_vectors = filtered_scene_flow_vecs
    
    # Convert to numpy
    sampled_pts_np = sampled_pts.cpu().numpy()
    sampled_vectors_np = sampled_vectors.cpu().numpy()
    
    # Calculate flow magnitudes
    flow_magnitudes = np.linalg.norm(sampled_vectors_np, axis=1)
    
    # Calculate bounds for magnitude if not provided
    if scene_bounds is None:
        if len(flow_magnitudes) == 0:
            return (0, 1)
        mag_min = flow_magnitudes.min()
        mag_max = flow_magnitudes.max()
        if mag_min == mag_max:
            mag_max = mag_min + 1e-6
    else:
        mag_min, mag_max = scene_bounds
    
    # Create colors based on flow magnitude and direction
    if len(flow_magnitudes) > 0 and mag_max > mag_min:
        # Normalize flow vectors for direction
        normalized_flow = sampled_vectors_np / (flow_magnitudes[:, np.newaxis] + 1e-8)
        
        # Hue from XZ plane angle
        hue = np.arctan2(normalized_flow[:, 2], normalized_flow[:, 0])
        hue = (hue + np.pi) / (2 * np.pi)
        
        # Saturation and value from magnitude
        normalized_magnitude = np.clip((flow_magnitudes - mag_min) / (mag_max - mag_min + 1e-8), 0, 1)
        saturation = 0.3 + 0.7 * normalized_magnitude
        value = 0.5 + 0.5 * normalized_magnitude
        
        # Convert HSV to RGB
        hsv = np.stack([hue, saturation, value], axis=1)
        colors = hsv_to_rgb(hsv)
    else:
        colors = np.ones((len(sampled_vectors_np), 3)) * 0.5
    
    # Log flow vectors as arrows
    rr.log(
        f"{base_name}/scene_flow",
        rr.Arrows3D(
            origins=sampled_pts_np,
            vectors=sampled_vectors_np,
            colors=colors,
        ),
    )

    return (mag_min, mag_max)


def log_point_tracks(point_tracks):
    """
    Log the point tracks for visualization.
    """
    # Create colors based on x-position of initial points
    initial_positions = point_tracks[0]
    x_coords = initial_positions[:, 0]
    x_min = x_coords.min()
    x_max = x_coords.max()
    
    norm = mpl.colors.Normalize(vmin=x_min, vmax=x_max)
    normalized_x = norm(x_coords)
    colormap = cm.get_cmap('rainbow')
    track_colors = colormap(normalized_x)[:, :3]  # RGB only


    for track_idx in range(point_tracks.shape[1]):
        track = point_tracks[:, track_idx, :]
        rr.log(
            f"pred/point_tracks/track_{track_idx}",
            rr.LineStrips3D(
                strips=[track],
                colors=[track_colors[track_idx]],
            )
        )


def convert_sceneflow_ego_to_allo(ego_sf, pts3d, camera_pose):
    """
    Convert ego-centric scene flow to allo-centric using the camera pose.
    
    Args:
        ego_sf: Ego-centric scene flow
        pts3d: 3D points
        pose1: Camera pose matrix
    
    Returns:
        Allo-centric scene flow
    """
    pts3d_ego = pts3d + ego_sf
    pts3d_allo = pts3d_ego @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    allo_sf = pts3d_allo - pts3d
    return allo_sf

def init_inference_model(config, ckpt_path, device):
    "Initialize the model for inference"
    # Load the model
    if isinstance(config, dict):
        config_path = config["path"]
        overrrides = config["config_overrides"]
        model_args = init_hydra_config(config_path, overrides=overrrides)
        model = init_model(model_args.model.model_str, model_args.model.model_config)
    else:
        config_path = config
        model_args = init_hydra_config(config_path)
        model = init_model(model_args.model_str, model_args.model_config)
    model.to(device)
    if ckpt_path is not None:
        print("Loading model from: ", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        model.to(device)
    # Set the model to eval mode
    model.eval()

    return model


@torch.no_grad()
def sample_inference(model, views, device, use_amp):
    # Run inference
    result = loss_of_one_batch_multi_view(
        views,
        model,
        None,
        device,
        use_amp=use_amp,
    )

    return result


def visualize_raw_custom_data_inference_output(args, views, pred_output, img_norm_type, use_scene_flow_type, start_view_idx=0):
    # Visualize the results
    viz_mask_0 = None
    depth_z_0 = None

    for view_idx, view in enumerate(views):
        image = rgb(view["img"], norm_type=img_norm_type)

        # Visualize the predicted pointmaps
        pts_name = f"pred/pointcloud_view_{view_idx+start_view_idx}"
        pts_key = "pts3d"
        pred_pts3d = pred_output[f"pred{view_idx+1}"][pts_key][0].cpu()
        pred_pts3d_0 = pred_output[f"pred1"]["pts3d"][0].cpu()

        # Get the non ambiguous class mask if available
        non_ambiguous_mask = view["non_ambiguous_mask"].cpu()

        # Calculate normal mask
        normals, normals_mask = points_to_normals(pred_pts3d.numpy(), mask=non_ambiguous_mask.numpy())
        normal_edges = normals_edge(normals, tol=5, mask=normals_mask)

        # Calculate depth mask
        depth_z = pred_output[f"pred{view_idx+1}"]["pts3d_cam"][...,2:3][0].squeeze(-1).cpu().numpy()
        depth_edges = depth_edge(depth_z, rtol=0.03, mask=non_ambiguous_mask.numpy())

        # Combine both edge types
        mask = ~(depth_edges & normal_edges)

        # Combine with non ambiguous mask
        mask = non_ambiguous_mask.numpy() & mask

        # Close Depth mask
        close_depth_mask = depth_z < 40.0
        mask = mask & close_depth_mask

        if view_idx == 0:
            viz_mask_0 = mask

            kernel_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            viz_mask_0 = viz_mask_0.astype(np.uint8) * 255  # Convert bool → uint8
            viz_mask_0 = cv2.erode(viz_mask_0, kernel, iterations=3)

            depth_z_0 = depth_z.copy()

        if "cam_quats" not in pred_output[f"pred{view_idx+1}"].keys():
            # Visualize the image
            base_name = f"pred/image_view_{view_idx+start_view_idx}"
            rr.log(
                f"{base_name}/pinhole/rgb",
                rr.Image(image),
            )
            # Visualize the pointmaps
            log_points_to_rerun(image[0], pred_pts3d, pts_name, mask=mask)
            # Visualize the mask if available
            if mask is not None:
                rr.log(
                    f"{base_name}/pinhole/mask",
                    rr.SegmentationImage(mask.numpy().astype(int)),
                )
        else:
            base_name = f"pred/image_view_{view_idx+start_view_idx}"
            cam_quats = pred_output[f"pred{view_idx+1}"]["cam_quats"][0].cpu()
            cam_trans = pred_output[f"pred{view_idx+1}"]["cam_trans"][0].cpu()
            ray_directions = pred_output[f"pred{view_idx+1}"]["ray_directions"][0].cpu()
            ray_depth = pred_output[f"pred{view_idx+1}"]["depth_along_ray"][0].cpu()
            local_pts3d = ray_directions * ray_depth
            depth_z = local_pts3d[..., 2:]
            if mask is not None:
                ambiguous_mask = ~mask
                depth_z[ambiguous_mask] = 0
                viz_mask = mask
            else:
                viz_mask = None

            if "motion_mask" in pred_output[f"pred{view_idx+1}"].keys() and view_idx == 0:
                pred_motion_mask = pred_output[f"pred{view_idx+1}"]["motion_mask"][0].cpu().numpy()
                viz_mask = viz_mask * pred_motion_mask

            cam_rot = quaternion_to_rotation_matrix(cam_quats)
            cam_pose = torch.eye(4)
            cam_pose[:3, :3] = cam_rot
            cam_pose[:3, 3] = cam_trans
            cam_intrinsics = recover_pinhole_intrinsics_from_ray_directions(ray_directions)
            log_data_to_rerun(
                image[0],
                depth_z,
                cam_pose,
                cam_intrinsics,
                pred_pts3d,
                mask,
                base_name,
                pts_name,
                viz_mask=viz_mask,
            )

            # Get second camera pose
            cam_quats_0 = pred_output[f"pred{1}"]["cam_quats"][0].cpu()
            cam_trans_0 = pred_output[f"pred{1}"]["cam_trans"][0].cpu()
            cam_rot_0 = quaternion_to_rotation_matrix(cam_quats_0)
            cam_pose_0 = torch.eye(4)
            cam_pose_0[:3, :3] = cam_rot_0
            cam_pose_0[:3, 3] = cam_trans_0

        if "scene_flow" in pred_output[f"pred{view_idx+1}"].keys() and view_idx > 0:
            if use_scene_flow_type == "allo_scene_flow":
                # Log the predicted scene flow vectors
                pred_scene_flow_vectors = pred_output[f"pred{view_idx+1}"]["scene_flow"][0].cpu()

                if args.use_scene_flow_mask_refined==True:
                    # Get dynamic scene flow mask from scene flow
                    motion_sf_mask = pred_scene_flow_vectors.norm(dim=-1) > 1e-1
                    scene_flow_mask = viz_mask_0 & motion_sf_mask.numpy() & views[0]["binary_mask"].cpu().numpy()
                    mask = (scene_flow_mask > 0)
                    depth_values = depth_z_0[mask]
                    if len(depth_values) > 0:
                        depth_mean = np.median(depth_values)
                        depth_std = np.std(depth_values)
                    else:
                        depth_mean, depth_std = 0, 0
                    depth_tolerance = 0.5 * depth_std  # you can tune: 0.5 for very strict, 1.5 for loose
                    depth_min = depth_mean - depth_tolerance
                    depth_max = depth_mean + depth_tolerance
                    depth_consistent_mask = np.logical_and(
                        mask,
                        np.logical_and(depth_z_0 >= depth_min, depth_z_0 <= depth_max)
                    )
                    scene_flow_mask_refined = depth_consistent_mask.astype(np.uint8) * 255
                    log_scene_flow_to_rerun(image[0], pred_pts3d_0, pred_scene_flow_vectors, f"pred/scene_flow_{view_idx+start_view_idx}", mask=scene_flow_mask_refined)
                    return scene_flow_mask_refined
                else:
                    scene_flow_mask = viz_mask_0 & views[0]["binary_mask"].cpu().numpy()
                    log_scene_flow_to_rerun(image[0], pred_pts3d_0, pred_scene_flow_vectors, f"pred/scene_flow_{view_idx+start_view_idx}", mask=scene_flow_mask)
                    return scene_flow_mask

def test_anymap_only_on_custom_multi_view_video_images(args, high_level_config, resolution, video_images_folder_path, use_scene_flow_type, ref_img_binary_mask_path):
    if args.viz:
        rr.script_setup(args, f"Custom_Sample_Inference_AnyMapMultiView")
        rr.connect_grpc(f"rerun+http://127.0.0.1:{args.port}/proxy", flush_timeout_sec=None)
        rr.set_time_seconds("stable_time", 0)
        rr.log("pred", rr.ViewCoordinates.RDF, static=True)

    # Initialize the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load the trained model
    trained_model = init_inference_model(high_level_config, high_level_config["checkpoint_path"], device)

    # Load the images
    image_paths = natsorted(glob(os.path.join(video_images_folder_path, "*.jpg")))

    if args.ref_img_idx and args.start_idx and args.end_idx is not None:
        start_idx = args.start_idx
        img_idx = args.ref_img_idx
        end_idx = args.end_idx
    else:
        start_idx = max(0, len(image_paths) // 2 - 20)
        img_idx = len(image_paths) // 2
        end_idx =  min(len(image_paths) // 2 + 20, len(image_paths))


    moge_model = load_moge_model(device="cuda")

    image_list = [image_paths[img_idx]]
    image_list += [image_paths[idx] for idx in range(start_idx, end_idx,1)]


    # Reconstruction between first and 'idx' image
    views = load_images(
        image_list,
        size=resolution,
        verbose=False,
        norm_type=high_level_config["data_norm_type"],
        patch_size=14,
        compute_moge_mask=True,
        moge_model=moge_model,
        binary_mask_path=ref_img_binary_mask_path,
    )

    # Run inference with trained model
    pred_result = sample_inference(
        trained_model,
        views,
        device,
        use_amp=high_level_config["trained_with_amp"],
    )

    if args.viz:
        viz_point_track_views = [views[0], views[10]]
        viz_point_track_pred_result = {}
        viz_point_track_pred_result["view1"] = pred_result["view1"]
        viz_point_track_pred_result["view2"] = pred_result["view11"]
        viz_point_track_pred_result["pred1"] = pred_result["pred1"]
        viz_point_track_pred_result["pred2"] = pred_result["pred11"]

        scene_flow_mask_refined = visualize_raw_custom_data_inference_output(
            args, viz_point_track_views, viz_point_track_pred_result, img_norm_type=high_level_config["data_norm_type"], use_scene_flow_type=use_scene_flow_type
        )


    point_tracks = []
    for idx in range(1,len(views)):
        cur_views = [views[0], views[idx]]
        cur_pred_result = {}
        cur_pred_result["view1"] = pred_result["view1"]
        cur_pred_result["view2"] = pred_result[f"view{idx+1}"]
        cur_pred_result["pred1"] = pred_result["pred1"]
        cur_pred_result["pred2"] = pred_result[f"pred{idx+1}"]
        # Visualize the results
        if args.viz:
            rr.set_time_seconds("stable_time", 0.1*idx)
            visualize_raw_custom_data_inference_output(
                args, cur_views, cur_pred_result, img_norm_type=high_level_config["data_norm_type"], use_scene_flow_type=use_scene_flow_type
            )

        if args.viz:
            rr.script_teardown(args)

        cur_pts3d = cur_pred_result["pred1"]["pts3d"][0].cpu()
        cur_scene_flow = cur_pred_result["pred2"]["scene_flow"][0].cpu()
        pts3d_after_motion = cur_pts3d + cur_scene_flow
        viz_mask = torch.tensor(scene_flow_mask_refined) > 0
        valid_points = pts3d_after_motion[viz_mask][::50]
        point_tracks.append(valid_points)

        if args.viz:
            log_point_tracks(np.asarray(point_tracks))

        if args.viz:
            rr.script_teardown(args)

def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--machine",type=str, default="local")
    parser.add_argument("--config_dir",type=str, default="configs")
    parser.add_argument("--checkpoint_path",type=str, default="checkpoints/any4d_4v_combined.pth")
    parser.add_argument("--video_images_folder_path", type=str, required=True)
    parser.add_argument("--ref_img_binary_mask_path", type=str, required=False, default=None)
    parser.add_argument("--start_idx", type=int, required=False, default=None)
    parser.add_argument("--end_idx", type=int, required=False, default=None)
    parser.add_argument("--ref_img_idx", type=int, required=False, default=None)
    parser.add_argument("--use_scene_flow_mask_refined", default=True)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--port", type=int, default=9876)

    return parser

if __name__ == "__main__":
    # Parser for Rerun
    parser = get_parser()
    script_add_rerun_args(parser)  # Options: --headless, --connect, --serve, --url, --save, --stdout
    args = parser.parse_args()

    # Set the seed
    seed_everything(0)

    # Experiment name
    experiment = "518"

    #Any4D model
    any4d_high_level_config = {
        "path": f"{args.config_dir}/train.yaml",
        "config_overrides": [
            f"machine={args.machine}",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ],
        "checkpoint_path": args.checkpoint_path,
        "trained_with_amp": True,
        "data_norm_type": "dinov2",
    }

    test_anymap_only_on_custom_multi_view_video_images(
        args,
        any4d_high_level_config,
        resolution=(518, 336),
        video_images_folder_path= args.video_images_folder_path, #"/home/jaykarhade/Downloads/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/stroller",
        use_scene_flow_type="allo_scene_flow",
        ref_img_binary_mask_path= args.ref_img_binary_mask_path #'/ocean/projects/cis220039p/mdt2/datasets/dydust3r/davis/masked_images/480p/stroller',
    )