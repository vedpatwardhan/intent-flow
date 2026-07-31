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
    depth_map = sim.renderer.render(depth=True)  # Shape [H, W]
    img_h, img_w = depth_map.shape

    # Clamp pixel bounds
    px = int(np.clip(pixel_x, 0, img_w - 1))
    py = int(np.clip(pixel_y, 0, img_h - 1))

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


def unproject_ui_annotations_to_3d(
    sim, ui_annotations: dict, camera_name: str = "world_center"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses user 2D UI annotations (vectors/crops) and unprojects starting effector link and
    target object positions to 3D world coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: (effector_start_3d, target_object_3d)
    """
    vectors = ui_annotations.get("vectors", [])
    crops = ui_annotations.get("crops", [])

    effector_3d = None
    target_3d = None

    if vectors:
        v = vectors[0]
        # v contains 'start' (u0, v0) and 'end' (u1, v1) in 224x224 coordinates
        start_x, start_y = v.get("start", [112, 112])
        end_x, end_y = v.get("end", [112, 112])

        effector_3d = unproject_pixel_to_world(
            sim, start_x, start_y, camera_name=camera_name
        )
        target_3d = unproject_pixel_to_world(sim, end_x, end_y, camera_name=camera_name)
    elif crops:
        c = crops[0]
        # c contains bbox coordinates [x, y, w, h]
        bbox = c.get("bbox", [100, 100, 24, 24])
        center_x = bbox[0] + bbox[2] / 2.0
        center_y = bbox[1] + bbox[3] / 2.0
        target_3d = unproject_pixel_to_world(
            sim, center_x, center_y, camera_name=camera_name
        )

    # Fallback to simulation body positions if annotations are unparseable
    if effector_3d is None:
        index_id = sim.model.body("R_index_tip_link").id
        effector_3d = sim.data.xpos[index_id].copy()

    if target_3d is None:
        cube_id = sim.model.body("cube").id
        target_3d = sim.data.xpos[cube_id].copy()

    return effector_3d, target_3d
