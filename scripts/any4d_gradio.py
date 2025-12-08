import os
import tempfile
import uuid
import traceback

import gradio as gr
import torch
import cv2
import numpy as np
from pathlib import Path
from glob import glob
from natsort import natsorted

import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun

import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb
from PIL import Image

from any4d.utils.image import load_images, rgb
from any4d.utils.misc import seed_everything
from any4d.utils.moge_inference import load_moge_model
from any4d.utils.inference import loss_of_one_batch_multi_view
from any4d.models import init_model
from any4d.utils.geometry import (
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
    normals_edge,
    depth_edge,
    points_to_normals,
)

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


class Any4DProcessor:
    """Handles Any4D processing and Rerun visualization"""
    
    def __init__(self, config_dir, checkpoint_path, machine="local"):
        self.config_dir = config_dir
        self.checkpoint_path = checkpoint_path
        self.machine = machine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.moge_model = None
        
        seed_everything(0)
        
    def initialize(self):
        """Load models"""        
        high_level_config = {
            "path": f"{self.config_dir}/train.yaml",
            "config_overrides": [
                f"machine={self.machine}",
                "model=any4d",
                "model.encoder.uses_torch_hub=false",
                "model/task=images_only",
            ],
            "checkpoint_path": self.checkpoint_path,
            "trained_with_amp": True,
            "data_norm_type": "dinov2",
        }
        
        self.model = init_inference_model(
            high_level_config,
            self.checkpoint_path,
            self.device
        )
        self.moge_model = load_moge_model(model_code_path="any4d/external/MoGe", device=self.device)
        self.high_level_config = high_level_config
        
        print("✓ Models initialized")
        
    def extract_frames_from_video(self, video_path, max_frames=100):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def process_video_streaming(
        self,
        recording_id: str,
        video_path: str,
        max_frames: int = 40,
        img_idx: int = 0,
        use_scene_flow_mask_refinement: bool = True,
        progress=gr.Progress()
    ):
        """
        Process video and stream results to embedded Rerun viewer.
        
        This function yields data incrementally to the Rerun viewer
        embedded in the Gradio interface.
        """        
        
        # Create recording stream
        rec = rr.RecordingStream(
            application_id="any4d_visualization",
            recording_id=recording_id
        )
        stream = rec.binary_stream()

        try:
            progress(0, desc="Extracting frames...")
            
            # Extract frames
            frames = self.extract_frames_from_video(video_path, max_frames=max_frames)
            
            # Validate img_idx
            if img_idx >= len(frames):
                raise gr.Error(f"Reference frame index ({img_idx}) must be less than number of frames ({len(frames)})")
            
            # Save to temp directory
            temp_dir = tempfile.mkdtemp()
            image_paths = []
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.jpg")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                image_paths.append(frame_path)
            
            progress(0.2, desc="Loading images...")
            
            # Select frame range - img_idx is now a user parameter
            start_idx = 0
            end_idx = len(image_paths)
            
            image_list = [image_paths[img_idx]]
            image_list += [image_paths[idx] for idx in range(start_idx, end_idx, 1)]
            
            # Load images
            views = load_images(
                image_list,
                # size=(width, height),
                verbose=True,
                norm_type="dinov2",
                patch_size=14,
                compute_moge_mask=True,
                moge_model=self.moge_model,
                binary_mask_path=None
            )
            
            progress(0.4, desc="Running inference...")
            
            # Run inference
            pred_result = sample_inference(
                self.model,
                views,
                self.device,
                use_amp=True
            )
            
            progress(0.6, desc="Creating visualization...")
            
            # Set up coordinate system
            rec.log("pred", rr.ViewCoordinates.RDF, static=True)
            
            # Create blueprint for better initial view
            blueprint = rrb.Blueprint(
                rrb.Spatial3DView(
                    origin="pred",
                    name="3D Scene",
                    background=[255, 255, 255],  # White color (RGB)
                    line_grid=rrb.archetypes.LineGrid3D(
                        visible=False,
                    ),
                ),
                collapse_panels=True,
            )
            rec.send_blueprint(blueprint)
            
            # Yield initial setup
            yield stream.read()
            
            # Visualize each frame pair
            num_views = len(views)
            for idx in range(1, num_views):
                progress_val = 0.6 + (0.4 * idx / (num_views - 1))
                progress(progress_val, desc=f"Visualizing frame {idx}/{num_views-1}")
                
                cur_views = [views[0], views[idx]]
                cur_pred_result = {
                    "view1": pred_result["view1"],
                    "view2": pred_result[f"view{idx+1}"],
                    "pred1": pred_result["pred1"],
                    "pred2": pred_result[f"pred{idx+1}"]
                }
                
                # Set time for animation
                rec.set_time_seconds("stable_time", 0.2*idx)

                self.log_visualization_data(
                    rec,
                    cur_views, 
                    cur_pred_result, 
                    img_norm_type=self.high_level_config["data_norm_type"], 
                    use_scene_flow_type="allo_scene_flow",
                    use_scene_flow_mask_refinement=use_scene_flow_mask_refinement,
                )

                # Yield data to viewer incrementally
                yield stream.read()
            
            progress(1.0, desc="Complete!")
            
            # Final yield to ensure all data is sent
            yield stream.read()
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise gr.Error(error_msg)

    def log_visualization_data(self, rec, views, pred_output, img_norm_type, use_scene_flow_type, use_scene_flow_mask_refinement=True, start_view_idx=0):
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
                rec.log(
                    f"{base_name}/pinhole/rgb",
                    rr.Image(image),
                )
                # Visualize the pointmaps
                log_points_to_rerun(image[0], pred_pts3d, pts_name, mask=mask)
                # Visualize the mask if available
                if mask is not None:
                    rec.log(
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
                self.log_data_to_rerun(
                    rec,
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

            if "scene_flow" in pred_output[f"pred{view_idx+1}"].keys():
                if use_scene_flow_type == "allo_scene_flow":
                    # Log the predicted scene flow vectors
                    pred_scene_flow_vectors = pred_output[f"pred{view_idx+1}"]["scene_flow"][0].cpu()

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
                    depth_tolerance = 0.5 * depth_std
                    depth_min = depth_mean - depth_tolerance
                    depth_max = depth_mean + depth_tolerance
                    depth_consistent_mask = np.logical_and(
                        mask,
                        np.logical_and(depth_z_0 >= depth_min, depth_z_0 <= depth_max)
                    )
                    scene_flow_mask_refined = depth_consistent_mask.astype(np.uint8) * 255

                    if view_idx == 1:
                        # Use refined mask or basic mask based on user preference
                        mask_to_use = scene_flow_mask_refined if use_scene_flow_mask_refinement else viz_mask_0
                        self.log_scene_flow_to_rerun(rec, image[0], pred_pts3d_0, pred_scene_flow_vectors, f"pred/scene_flow_{view_idx+start_view_idx}", mask=mask_to_use)

        return scene_flow_mask_refined

    def log_data_to_rerun(self, rec, image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None):
        # Log camera info and loaded data
        height, width = image.shape[0], image.shape[1]
        rec.log(
            base_name,
            rr.Transform3D(
                translation=pose[:3, 3],
                mat3x3=pose[:3, :3],
                from_parent=False,
            ),
        )
        rec.log(
            f"{base_name}/pinhole",
            rr.Pinhole(
                image_from_camera=intrinsics,
                height=height,
                width=width,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
        rec.log(
            f"{base_name}/pinhole/rgb",
            rr.Image(image),
        )
        # Log points in 3D
        filtered_pts = pts3d[mask]
        filtered_pts_col = image[mask]
        rec.log(
            pts_name,
            rr.Points3D(
                positions=filtered_pts.reshape(-1, 3),
                colors=filtered_pts_col.reshape(-1, 3),
            ),
        )

    def log_scene_flow_to_rerun(self, rec, image, pts3d, scene_flow_vecs, base_name, mask=None, scene_bounds=None, colormap_name='rainbow'):
        """
        Log scene flow in 3D with color visualization based on flow magnitude and direction
        """
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
        rec.log(
            f"{base_name}/scene_flow",
            rr.Arrows3D(
                origins=sampled_pts_np,
                vectors=sampled_vectors_np,
                colors=colors,
            ),
        )

        return (mag_min, mag_max)


def create_app(config_dir, checkpoint_path):
    """Create Gradio app with embedded Rerun viewer"""
    
    # Initialize processor
    processor = Any4DProcessor(config_dir, checkpoint_path)
    processor.initialize()
    
    # Create Gradio interface
    with gr.Blocks(
        title="Any4D Scene Flow Visualization",
    ) as demo:
        
        gr.Markdown("""
        # 🎬 Any4D Scene Flow Visualization
        
        Upload a video to visualize 3D reconstruction and scene flow in real-time.
        The Rerun viewer is embedded directly in this interface!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                
                video_input = gr.Video(
                    label="Upload Video",
                    sources=["upload"]
                )
                
                with gr.Row():
                    max_frames_slider = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=40,
                        step=1,
                        label="Max Frames to Extract",
                        info="Number of frames to extract from video"
                    )
                
                # Frame preview section
                with gr.Row():
                    frame_preview = gr.Image(
                        label="Reference Frame Preview",
                        type="numpy",
                        interactive=False
                    )
                
                with gr.Row():
                    reference_frame_slider = gr.Slider(
                        minimum=0,
                        maximum=39,
                        value=20,
                        step=1,
                        label="Reference Frame Index",
                        info="Select which frame to use as reference for scene flow"
                    )
                
                with gr.Row():
                    use_mask_refinement = gr.Checkbox(
                        value=False,
                        label="Use Scene Flow Mask Refinement",
                        info=" Removes background scene-flow smearing near boundaries by assuming dominant motion in scene. Do not use for scenes with multiple dynamic objects."
                    )
                
                process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                
                status_text = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
            
            with gr.Column(scale=3):
                gr.Markdown("### 🎥 Live 3D Visualization")
                
                # Embedded Rerun viewer with streaming disabled when running locally (on HF it is enabled)
                viewer = Rerun(
                    streaming=False,
                    height=1000,
                    panel_states={
                        "time": "collapsed",
                        "blueprint": "collapsed",
                        "selection": "collapsed",
                    },
                )
        
        # Store recording ID and extracted frames in session state
        recording_id = gr.State(lambda: str(uuid.uuid4()))
        extracted_frames = gr.State([])
        
        # Function to extract and preview frames
        def extract_and_preview(video_file, max_frames, progress=gr.Progress()):
            if video_file is None:
                return None, gr.update(maximum=0, value=0), []
            
            try:
                progress(0, desc="Extracting frames...")
                
                # Extract frames
                cap = cv2.VideoCapture(video_file)
                frames = []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                interval = max(1, total_frames // max_frames)
                
                frame_count = 0
                while cap.isOpened() and len(frames) < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                    
                    frame_count += 1
                
                cap.release()
                
                num_frames = len(frames)
                
                # Set default to middle frame
                default_idx = num_frames // 2 if num_frames > 0 else 0
                
                # Return middle frame as preview and update slider
                return (
                    frames[default_idx] if num_frames > 0 else None,
                    gr.update(maximum=max(0, num_frames - 1), value=default_idx),
                    frames
                )
                
            except Exception as e:
                print(f"Error extracting frames: {e}")
                return None, gr.update(maximum=0, value=0), []
        
        # Function to update frame preview when slider changes
        def update_frame_preview(frames, frame_idx):
            if len(frames) == 0 or frame_idx >= len(frames):
                return None
            return frames[frame_idx]
        
        # Processing function
        def process_video_wrapper(video_file, max_frames, ref_frame_idx, use_mask_ref, rec_id, progress=gr.Progress()):
            if video_file is None:
                raise gr.Error("Please upload a video first")
            
            try:
                # Stream data to embedded viewer
                for stream_data in processor.process_video_streaming(
                    rec_id,
                    video_file,
                    max_frames,
                    ref_frame_idx,
                    use_mask_ref,
                    progress
                ):
                    # Yield both the stream data AND a status message
                    yield stream_data, "🔄 Processing... Please wait."
                
                # Final yield with completion message
                mask_status = "with refinement" if use_mask_ref else "without refinement"
                yield stream_data, f"✅ Processing complete! Reference frame: {ref_frame_idx}, Scene flow mask {mask_status}. Explore the 3D visualization above."
                
            except Exception as e:
                # On error, yield None for viewer and error message for status
                yield None, f"❌ Error: {str(e)}"

        # Connect video upload to frame extraction
        video_input.change(
            extract_and_preview,
            inputs=[video_input, max_frames_slider],
            outputs=[frame_preview, reference_frame_slider, extracted_frames]
        )
        
        # Update max_frames and re-extract when slider changes
        max_frames_slider.change(
            extract_and_preview,
            inputs=[video_input, max_frames_slider],
            outputs=[frame_preview, reference_frame_slider, extracted_frames]
        )
        
        # Update preview when reference frame slider changes
        reference_frame_slider.change(
            update_frame_preview,
            inputs=[extracted_frames, reference_frame_slider],
            outputs=[frame_preview]
        )
        
        # Connect button to processing
        process_btn.click(
            process_video_wrapper,
            inputs=[video_input, max_frames_slider, reference_frame_slider, use_mask_refinement, recording_id],
            outputs=[viewer, status_text]
        )

        gr.Markdown("""
        ---
        ### 🎨 Visualization Guide
        
        **What you see:**
        - **3D Point Clouds**: Dense reconstruction colored by RGB
        - **Camera Trajectory**: Camera poses at each frame
        - **Scene Flow Vectors**: Colored arrows showing 3D motion
        - **Timeline**: Navigate through time using the timeline at the bottom
        
        **Controls:**
        - **Max Frames**: Control how many frames to extract from the video (more frames = slower but more detailed)
        - **Reference Frame**: Choose which frame serves as the reference for scene flow computation
        - **Frame Preview**: See the selected reference frame before processing
        - **Scene Flow Mask Refinement**: When enabled, applies depth-based filtering to produce cleaner scene flow (more conservative). Disable for more complete but potentially noisier results.
        
        **Tips:**
        - The visualization updates in real-time as processing happens
        - Use the timeline to see motion over time
        - Zoom in to see fine details of the reconstruction
        - Try different reference frames to see scene flow from different perspectives
        - Toggle mask refinement to see the difference in scene flow quality vs coverage
        - Best results with videos showing clear motion (2-10 seconds)
        """)
    
    return demo


if __name__ == "__main__":
    # Your actual paths
    CONFIG_DIR = "configs"
    CHECKPOINT_PATH = "checkpoints/any4d_4v_combined.pth"
    
    app = create_app(CONFIG_DIR, CHECKPOINT_PATH)
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )