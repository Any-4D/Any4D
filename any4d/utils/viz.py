"""
Utility functions for visualization
"""

from argparse import ArgumentParser, Namespace
from distutils.util import strtobool

import rerun as rr


def log_data_to_rerun(image, depthmap, pose, intrinsics, base_name, mask=None):
    """
    Log camera and image data to Rerun visualization tool.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image to be logged
    depthmap : numpy.ndarray
        Depth map corresponding to the image
    pose : numpy.ndarray
        4x4 camera pose matrix with rotation (3x3) and translation (3x1)
    intrinsics : numpy.ndarray
        Camera intrinsic matrix
    base_name : str
        Base name for the logged entities in Rerun
    mask : numpy.ndarray, optional
        Optional segmentation mask for the depth image
    """
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
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
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    if mask is not None:
        rr.log(
            f"{base_name}/pinhole/depth_mask",
            rr.SegmentationImage(mask),
        )


def str2bool(v):
    return bool(strtobool(v))


def script_add_rerun_args(parser: ArgumentParser) -> None:
    """
    Add common Rerun script arguments to `parser`.

    Change Log from https://github.com/rerun-io/rerun/blob/29eb8954b08e59ff96943dc0677f46f7ea4ea734/rerun_py/rerun_sdk/rerun/script_helpers.py#L65:
        - Added default portforwarding url for ease of use
        - Update parser types

    Parameters
    ----------
    parser : ArgumentParser
        The parser to add arguments to.

    Returns
    -------
    None
    """
    parser.add_argument(
        "--headless",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Don't show GUI",
    )
    parser.add_argument(
        "--connect",
        dest="connect",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Connect to an external viewer",
    )
    parser.add_argument(
        "--serve",
        dest="serve",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Serve a web viewer (WARNING: experimental feature)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="rerun+http://127.0.0.1:2004/proxy",
        help="Connect to this HTTP(S) URL",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save data to a .rrd file at this path"
    )
    parser.add_argument(
        "-o",
        "--stdout",
        dest="stdout",
        action="store_true",
        help="Log data to standard output, to be piped into a Rerun Viewer",
    )


def init_rerun_args(
    headless=True,
    connect=True,
    serve=False,
    url="rerun+http://127.0.0.1:2004/proxy",
    save=None,
    stdout=False,
) -> Namespace:
    """
    Initialize common Rerun script arguments.

    Parameters
    ----------
    headless : bool, optional
        Don't show GUI, by default True
    connect : bool, optional
        Connect to an external viewer, by default True
    serve : bool, optional
        Serve a web viewer (WARNING: experimental feature), by default False
    url : str, optional
        Connect to this HTTP(S) URL, by default rerun+http://127.0.0.1:2004/proxy
    save : str, optional
        Save data to a .rrd file at this path, by default None
    stdout : bool, optional
        Log data to standard output, to be piped into a Rerun Viewer, by default False

    Returns
    -------
    Namespace
        The parsed arguments.
    """
    rerun_args = Namespace()
    rerun_args.headless = headless
    rerun_args.connect = connect
    rerun_args.serve = serve
    rerun_args.url = url
    rerun_args.save = save
    rerun_args.stdout = stdout

    return rerun_args
