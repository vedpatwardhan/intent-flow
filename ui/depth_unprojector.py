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


def extract_region_2d_extremes(
    sim,
    mask_224: list | np.ndarray,
    target_pt: tuple[float, float],
    other_pt: tuple[float, float] | None = None,
    camera_name: str = "world_center",
) -> list[tuple[float, float]]:
    """
    Extracts the 4 2D extreme boundary pixels (x_min, x_max, y_min, y_max) from a SAM combined_mask_224
    in 224x224 coordinate space for the object cluster closest to target_pt. Pre-filters mask pixels
    against background depth BEFORE extracting extreme boundary points so boundary background pixels
    never corrupt extreme boundaries.
    """
    if mask_224 is None:
        raise ValueError(
            "❌ [Annotation Error] combined_mask_224 in task_isolated_features cannot be None."
        )

    mask_arr = np.array(mask_224) if not isinstance(mask_224, np.ndarray) else mask_224

    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        raise ValueError(
            "❌ [Annotation Error] SAM combined_mask_224 is empty (no activated mask pixels)."
        )

    pts = np.column_stack((xs, ys))  # Shape [N, 2] in (x, y) 224 space

    # 1. Partition mask pixels if both start and end points are provided
    tx, ty = target_pt
    if other_pt is not None:
        ox, oy = other_pt
        d_target = (pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2
        d_other = (pts[:, 0] - ox) ** 2 + (pts[:, 1] - oy) ** 2
        pts = pts[d_target < d_other]

    # 2. Filter mask pixels to object cluster near target_pt (within 60 pixels radius)
    if len(pts) > 0:
        dists = (pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2
        close_mask = dists <= (60.0**2)
        if np.any(close_mask):
            pts = pts[close_mask]

    if len(pts) == 0:
        return [(tx, ty), (tx, ty), (tx, ty), (tx, ty)]

    # 3. Unproject 3D target centroid to test 3D workspace foreground depth
    target_3d = unproject_pixel_to_world(sim, tx, ty, camera_name=camera_name)

    # 4. PRE-FILTER MASK PIXELS: Filter out any mask pixel whose unprojected 3D distance
    # from target_3d centroid exceeds workspace object radius (25cm) or hits background plane
    valid_pts = []
    for px, py in pts:
        p3d = unproject_pixel_to_world(
            sim, float(px), float(py), camera_name=camera_name
        )
        if np.linalg.norm(p3d - target_3d) <= 0.25:
            valid_pts.append((float(px), float(py)))

    if not valid_pts:
        return [(tx, ty), (tx, ty), (tx, ty), (tx, ty)]

    valid_pts_arr = np.array(valid_pts)  # [M, 2]

    x_min_idx = np.argmin(valid_pts_arr[:, 0])
    x_max_idx = np.argmax(valid_pts_arr[:, 0])
    y_min_idx = np.argmin(valid_pts_arr[:, 1])
    y_max_idx = np.argmax(valid_pts_arr[:, 1])

    return [
        (float(valid_pts_arr[x_min_idx, 0]), float(valid_pts_arr[x_min_idx, 1])),
        (float(valid_pts_arr[x_max_idx, 0]), float(valid_pts_arr[x_max_idx, 1])),
        (float(valid_pts_arr[y_min_idx, 0]), float(valid_pts_arr[y_min_idx, 1])),
        (float(valid_pts_arr[y_max_idx, 0]), float(valid_pts_arr[y_max_idx, 1])),
    ]


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
    sim,
    ui_annotations: dict,
    task_isolated_features: dict,
    camera_name: str = "world_center",
) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """
    Strictly parses user 2D annotations (vectors) and SAM combined_mask_224 from task_isolated_features
    by pre-filtering valid foreground mask pixels, extracting 4 2D extreme boundary pixels, unprojecting
    them to 3D world space, resolving the robot link, and returning 3D bounding extents.

    Returns:
        tuple[np.ndarray, np.ndarray, dict, str]:
            (effector_start_3d, target_object_3d, target_3d_bounds, selected_body_name)
    """
    if not task_isolated_features or not isinstance(task_isolated_features, dict):
        raise ValueError(
            "❌ [Annotation Error] task_isolated_features dictionary is required and cannot be None."
        )

    combined_mask_224 = task_isolated_features.get("combined_mask_224")
    if combined_mask_224 is None:
        raise ValueError(
            "❌ [Annotation Error] task_isolated_features['combined_mask_224'] is missing or invalid."
        )

    vectors = ui_annotations.get("vectors", [])
    if not vectors:
        raise ValueError(
            f"❌ [Annotation Error] No 2D intent vectors found in annotations for view '{camera_name}'."
        )

    v = vectors[0]
    if "start" not in v or "end" not in v:
        raise ValueError(
            f"❌ [Annotation Error] Invalid vector structure in annotations: {v}. Must contain 'start' and 'end'."
        )

    start_x, start_y = float(v["start"][0]), float(v["start"][1])
    end_x, end_y = float(v["end"][0]), float(v["end"][1])

    # 1. Unproject vector start and end centroids to 3D world space
    effector_3d = unproject_pixel_to_world(
        sim, start_x, start_y, camera_name=camera_name
    )
    target_3d = unproject_pixel_to_world(sim, end_x, end_y, camera_name=camera_name)

    # 2. Extract 2D extreme boundary pixels PRE-FILTERED against background depth
    effector_2d_extremes = extract_region_2d_extremes(
        sim,
        combined_mask_224,
        target_pt=(start_x, start_y),
        other_pt=(end_x, end_y),
        camera_name=camera_name,
    )
    target_2d_extremes = extract_region_2d_extremes(
        sim,
        combined_mask_224,
        target_pt=(end_x, end_y),
        other_pt=(start_x, start_y),
        camera_name=camera_name,
    )

    # 3. Unproject pre-filtered 2D extremes to 3D world space
    effector_3d_extremes = [
        unproject_pixel_to_world(sim, px, py, camera_name=camera_name)
        for (px, py) in effector_2d_extremes
    ]
    target_3d_extremes = [
        unproject_pixel_to_world(sim, px, py, camera_name=camera_name)
        for (px, py) in target_2d_extremes
    ]

    target_3d_pts = np.array(target_3d_extremes)  # [4, 3]

    extents_raw = target_3d_pts.max(axis=0) - target_3d_pts.min(axis=0)

    target_3d_bounds = {
        "min_3d": target_3d_pts.min(axis=0).tolist(),
        "max_3d": target_3d_pts.max(axis=0).tolist(),
        "center_3d": target_3d.tolist(),
        "extents_3d": extents_raw.tolist(),
    }

    # 4. Dynamically find closest robot body link across all 4 3D effector extreme points
    selected_body_name = find_nearest_robot_body_multi(sim, effector_3d_extremes)

    return effector_3d, target_3d, target_3d_bounds, selected_body_name
