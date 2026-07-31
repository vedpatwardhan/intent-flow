import numpy as np
import mujoco


def unproject_pixel_to_world(
    sim, pixel_x: float, pixel_y: float, camera_name: str = "world_center"
) -> np.ndarray:
    """
    Unprojects a 2D image pixel coordinate (u, v) from camera space to 3D world coordinates (X_w, Y_w, Z_w).

    Parameters:
        sim: The simulation wrapper containing MuJoCo model, data, and renderer.
        pixel_x (float): Horizontal pixel coordinate u in range [0, img_width].
        pixel_y (float): Vertical pixel coordinate v in range [0, img_height].
        camera_name (str): Camera identifier name.

    Returns:
        np.ndarray: 3D world space position vector [X_w, Y_w, Z_w].
    """
    # 1. Update scene and render depth map
    sim.renderer.update_scene(sim.data, camera=camera_name)
    sim.renderer.enable_depth_rendering()
    depth_map = (
        sim.renderer.render().copy()
    )  # Shape [H, W], float32 depth values in meters
    sim.renderer.disable_depth_rendering()
    img_h, img_w = depth_map.shape

    # Scale 2D pixel coordinates from UI 224x224 space to actual depth_map dimensions
    scale_x = img_w / 224.0
    scale_y = img_h / 224.0
    scaled_pixel_x = pixel_x * scale_x
    scaled_pixel_y = pixel_y * scale_y

    # Clamp pixel bounds
    px = int(np.clip(scaled_pixel_x, 0, img_w - 1))
    py = int(np.clip(scaled_pixel_y, 0, img_h - 1))

    # Retrieve orthogonal camera depth Z_c
    depth_val = float(depth_map[py, px])

    # 2. Extract camera parameters from MuJoCo model
    cam_id = sim.model.camera(camera_name).id
    fovy = sim.model.cam_fovy[cam_id]  # Vertical FOV in degrees

    # Compute focal length and principal point
    f_y = (0.5 * img_h) / np.tan(np.radians(fovy / 2.0))
    f_x = f_y  # Square pixel assumption
    c_x = img_w / 2.0
    c_y = img_h / 2.0

    # Perspective unprojection to 3D Camera Frame (X_c, Y_c, Z_c)
    # MuJoCo camera coordinate convention: +X right, +Y up, -Z optical axis looking forward
    x_c = (px - c_x) * depth_val / f_x
    y_c = -(py - c_y) * depth_val / f_y  # Flip y for top-down image coordinates
    z_c = -depth_val

    p_cam = np.array([x_c, y_c, z_c], dtype=np.float32)

    # 3. Retrieve Camera Extrinsics (Position & Rotation Matrix)
    cam_xpos = sim.data.cam_xpos[cam_id]  # 3D position in world [3]
    cam_xmat = sim.data.cam_xmat[cam_id].reshape(3, 3)  # Rotation matrix [3, 3]

    # Rigid transformation to 3D World Frame: P_w = R * P_c + t
    p_world = cam_xmat @ p_cam + cam_xpos
    return p_world


def extract_region_2d_extremes(anno: dict) -> list[tuple[float, float]]:
    """
    Extracts the 4 2D extreme boundary pixels (x_min, x_max, y_min, y_max) from a segment or crop annotation.
    """
    if (
        "width" in anno
        and "height" in anno
        and anno["width"] > 0
        and anno["height"] > 0
    ):  # Crop Bounding Box
        x1, y1 = anno["x"], anno["y"]
        w, h = anno["width"], anno["height"]
        x2, y2 = x1 + w, y1 + h
        x_mid, y_mid = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return [(x1, y_mid), (x2, y_mid), (x_mid, y1), (x_mid, y2)]

    if "boundingBox" in anno:
        bbox = anno["boundingBox"]
        x1, y1, w, h = (
            bbox.get("x", 0),
            bbox.get("y", 0),
            bbox.get("width", 0),
            bbox.get("height", 0),
        )
        if w > 0 and h > 0:
            x2, y2 = x1 + w, y1 + h
            x_mid, y_mid = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return [(x1, y_mid), (x2, y_mid), (x_mid, y1), (x_mid, y2)]

    # Segment Point Cloud / Mask / Path
    points = (
        anno.get("points")
        or anno.get("path")
        or anno.get("contour")
        or anno.get("polygon")
    )
    if points and len(points) > 0:
        pts = np.array(points)  # Shape [N, 2]
        if pts.ndim == 2 and pts.shape[1] >= 2:
            x_min_idx = np.argmin(pts[:, 0])
            x_max_idx = np.argmax(pts[:, 0])
            y_min_idx = np.argmin(pts[:, 1])
            y_max_idx = np.argmax(pts[:, 1])
            return [
                (float(pts[x_min_idx, 0]), float(pts[x_min_idx, 1])),
                (float(pts[x_max_idx, 0]), float(pts[x_max_idx, 1])),
                (float(pts[y_min_idx, 0]), float(pts[y_min_idx, 1])),
                (float(pts[y_max_idx, 0]), float(pts[y_max_idx, 1])),
            ]

    # Single Point Fallback
    cx, cy = float(anno.get("x", 112)), float(anno.get("y", 112))
    return [(cx, cy), (cx, cy), (cx, cy), (cx, cy)]


def find_nearest_robot_body_multi(sim, effector_3d_extremes: list[np.ndarray]) -> str:
    """
    Finds the robot body/link in the MuJoCo simulation model that minimizes the aggregate
    Euclidean distance across all 4 unprojected 3D extreme points of the effector region.
    """
    min_dist = float("inf")
    closest_body_name = None

    for i in range(sim.model.nbody):
        body_name = sim.model.body(i).name
        if any(
            skip in body_name.lower()
            for skip in ["world", "ground", "table", "cube", "floor", "pedestal"]
        ):
            continue

        body_pos = sim.data.xpos[i]
        aggregate_dist = sum(
            float(np.linalg.norm(body_pos - p3d)) for p3d in effector_3d_extremes
        )
        if aggregate_dist < min_dist:
            min_dist = aggregate_dist
            closest_body_name = body_name

    if closest_body_name is None:
        raise RuntimeError(
            "❌ [Robot Link Resolution Error] Unable to resolve any valid robot "
            "body link matching 3D start coordinates."
        )

    return closest_body_name


def unproject_ui_annotations_to_3d(
    sim, ui_annotations: dict, camera_name: str = "world_center"
) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """
    Strictly parses user 2D annotations (vectors, crops, segments) by matching vector
    start/end points to segments/crops, extracting 4 2D extreme boundary pixels,
    unprojecting them to 3D world space, resolving the robot link, and returning 3D
    bounding extents.

    Returns:
        tuple[np.ndarray, np.ndarray, dict, str]:
            (effector_start_3d, target_object_3d, target_3d_bounds, selected_body_name)
    """
    vectors = ui_annotations.get("vectors", [])
    if not vectors:
        raise ValueError(
            f"❌ [Annotation Error] No 2D intent vectors found in "
            "annotations for view '{camera_name}'."
        )

    v = vectors[0]
    if "start" not in v or "end" not in v:
        raise ValueError(
            "❌ [Annotation Error] Invalid vector structure in "
            f"annotations: {v}. Must contain 'start' and 'end'."
        )

    start_x, start_y = v["start"][0], v["start"][1]
    end_x, end_y = v["end"][0], v["end"][1]

    # 1. Unproject vector start and end centroids to 3D world space
    effector_3d = unproject_pixel_to_world(
        sim, start_x, start_y, camera_name=camera_name
    )
    target_3d = unproject_pixel_to_world(sim, end_x, end_y, camera_name=camera_name)

    # 2. Match vector start/end to active crops/segments
    active_regions = ui_annotations.get("crops", []) + ui_annotations.get(
        "segments", []
    )

    start_region = None
    end_region = None

    if active_regions:

        def get_region_centroid(reg: dict) -> tuple[float, float]:
            if "width" in reg and "height" in reg:
                return (
                    reg.get("x", 0) + reg["width"] / 2.0,
                    reg.get("y", 0) + reg["height"] / 2.0,
                )
            elif "points" in reg and reg["points"]:
                pts = np.array(reg["points"])
                return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            elif "center" in reg:
                return (reg["center"][0], reg["center"][1])
            return (reg.get("x", 0), reg.get("y", 0))

        # Find region closest to vector start
        start_dists = [
            (get_region_centroid(reg)[0] - start_x) ** 2
            + (get_region_centroid(reg)[1] - start_y) ** 2
            for reg in active_regions
        ]
        start_region = active_regions[int(np.argmin(start_dists))]

        # Find region closest to vector end
        end_dists = [
            (get_region_centroid(reg)[0] - end_x) ** 2
            + (get_region_centroid(reg)[1] - end_y) ** 2
            for reg in active_regions
        ]
        end_region = active_regions[int(np.argmin(end_dists))]

    # Extract 2D extreme boundary pixels
    effector_2d_extremes = (
        extract_region_2d_extremes(start_region)
        if start_region
        else [(start_x, start_y)] * 4
    )
    target_2d_extremes = (
        extract_region_2d_extremes(end_region) if end_region else [(end_x, end_y)] * 4
    )

    # 3. Unproject 2D extremes to 3D world space
    effector_3d_extremes = [
        unproject_pixel_to_world(sim, px, py, camera_name=camera_name)
        for (px, py) in effector_2d_extremes
    ]
    target_3d_extremes = [
        unproject_pixel_to_world(sim, px, py, camera_name=camera_name)
        for (px, py) in target_2d_extremes
    ]

    # Calculate 3D bounding extents for target object
    target_3d_pts = np.array(target_3d_extremes)  # [4, 3]
    target_3d_bounds = {
        "min_3d": target_3d_pts.min(axis=0).tolist(),
        "max_3d": target_3d_pts.max(axis=0).tolist(),
        "center_3d": target_3d.tolist(),
        "extents_3d": (target_3d_pts.max(axis=0) - target_3d_pts.min(axis=0)).tolist(),
    }

    # 4. Dynamically find closest robot body link across all 4 3D effector extreme points
    selected_body_name = find_nearest_robot_body_multi(sim, effector_3d_extremes)

    return effector_3d, target_3d, target_3d_bounds, selected_body_name
