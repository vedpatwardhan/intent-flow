import numpy as np
import mujoco
import os
import time
import datetime
import warnings
from pathlib import Path
from PIL import Image
import mink

from gr1_config import (
    COMPACT_WIRE_JOINTS,
    JOINT_LIMITS_MIN,
    JOINT_LIMITS_MAX,
    SCENE_PATH,
    FROZEN_JOINTS,
    IK_POSTURE_LOCKS,
)
from gr1_protocol import StandardScaler

# Suppress performance warnings from qpsolvers
warnings.filterwarnings("ignore", category=UserWarning, module="qpsolvers")


class GR1MuJoCoBase:
    """
    Shared Physical Foundation for GR-1 MuJoCo Simulations.
    Handles XML loading, IK solving, State extraction, and Perception.
    """

    def __init__(self, scene_path=SCENE_PATH, restrict_ik=True):
        print(f"--- GR-1 MODULAR BASE (MuJoCo) ---")
        self.restrict_ik = restrict_ik
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Perception & Cameras
        self.cam_names = [
            "world_top",
            "world_left",
            "world_right",
            "world_center",
            "world_wrist",
        ]
        self.frame_indices = {cam: 0 for cam in self.cam_names}

        # Renderer
        self.res = (224, 224)
        self.renderer = mujoco.Renderer(
            self.model, height=self.res[1], width=self.res[0]
        )

        # Mapping Protocol Names -> Joint IDs
        self.wire_min = np.array(JOINT_LIMITS_MIN)
        self.wire_max = np.array(JOINT_LIMITS_MAX)
        self._init_joint_mappings()
        self._init_finger_coupling()

        # Diagnostic Logging
        self.debug_log_path = None

        # Canonical Scaling Logic
        self.unscaler = StandardScaler()

        # IK Setup (Mink 1.1.0)
        self._init_ik_solver()

        # Internal state
        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_q_idx = self.model.jnt_qposadr[root_id]
        self.data.qpos[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]

        self.last_target_q = self.data.qpos.copy()
        self.active_joints_this_command = set()
        self.is_recording = False
        self.current_phase = 0  # 0: Neutral, 1: Approach, 2: Descent, 3: Grasp, 4: Lift
        self.rerun_count = 0
        self.render_step_idx = 0
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_log_dir = (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "temp_images"
            / self.session_id
        )
        self._init_joint_mappings()
        self._init_finger_coupling()

    def _init_joint_mappings(self):
        self.ik_names = set()
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Load IK whitelist
        i_path = os.path.join(base_path, "ik_joints.txt")
        with open(i_path, "r") as f:
            self.ik_names.update(
                [l.strip().split("#")[0].strip() for l in f if l.strip()]
            )

        print(f"✅ Loaded {len(self.ik_names)} IK joint names.")

        self.protocol_joint_ids = []
        self.v_allowed_mask = np.zeros(32)
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            try:
                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.protocol_joint_ids.append(j_id)
                if j_id != -1:
                    self.v_allowed_mask[i] = 1.0
            except:
                self.protocol_joint_ids.append(-1)

    def _init_finger_coupling(self):
        self.coupling_map = {}
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            if "proximal" in name.lower():
                base_prefix = name.split("_proximal")[0]
                for j in range(self.model.njnt):
                    j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    if j_name and base_prefix in j_name and j_name != name:
                        if "thumb" in name.lower() and (
                            ("yaw" in name.lower() and "pitch" in j_name.lower())
                            or ("pitch" in name.lower() and "yaw" in j_name.lower())
                        ):
                            continue
                        if i not in self.coupling_map:
                            self.coupling_map[i] = []
                        q_idx = self.model.jnt_qposadr[j]
                        self.coupling_map[i].append(q_idx)

    def _init_ik_solver(self):
        self.ee_index_link = "R_index_tip_link"
        self.ee_thumb_link = "R_thumb_tip_link"
        self.ee_wrist_link = "right_hand_pitch_link"
        self.configuration = mink.Configuration(self.model)

        # Determine authorized velocity indices (DOFs) for the IK Solver
        self.auth_dofs = set()
        whitelist = self.ik_names if self.restrict_ik else COMPACT_WIRE_JOINTS
        for name in whitelist:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                v_idx = self.model.jnt_dofadr[j_id]
                dof_counts = [6, 3, 1, 1]
                j_type = self.model.jnt_type[j_id]
                dnum = dof_counts[j_type]
                for d in range(dnum):
                    self.auth_dofs.add(int(v_idx + d))
        print(f"IK auth_dofs (restricted={self.restrict_ik}):", self.auth_dofs)

        all_dofs = set(range(self.model.nv))
        frozen_dofs = list(all_dofs - self.auth_dofs)

        self.tasks = [
            mink.FrameTask(
                frame_name=self.ee_index_link,
                frame_type="body",
                position_cost=5.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.FrameTask(
                frame_name=self.ee_thumb_link,
                frame_type="body",
                position_cost=5.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.FrameTask(
                frame_name=self.ee_wrist_link,
                frame_type="body",
                position_cost=2.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.PostureTask(model=self.model, cost=1e-6),
            mink.DofFreezingTask(model=self.model, dof_indices=frozen_dofs),
        ]
        self.tasks[3].set_target(self.model.qpos0)

        self.limits = [
            mink.ConfigurationLimit(model=self.model, min_distance_from_limits=0.01),
            mink.VelocityLimit(model=self.model),
        ]

    def _debug_log(self, msg):
        if self.debug_log_path is None:
            return
        t_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.debug_log_path, "a") as f:
            f.write(f"[{t_str}] {msg}\n")

    def _render_needs_depth(self) -> bool:
        return False

    def _post_render_hook(self, name, rgb, depth=None):
        pass

    def get_state_32(self):
        return self.qpos_to_action_32(self.data.qpos)

    def qpos_to_action_32(self, qpos):
        state = np.zeros(32, dtype=np.float32)
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1:
                q_idx = self.model.jnt_qposadr[j_id]
                state[i] = qpos[q_idx]
        return state

    def solve_ik(
        self,
        pos_wrist,
        quat,
        pos_index=None,
        pos_thumb=None,
        posture_target=None,
        posture_cost=None,
    ):
        quat = np.array(quat)
        if quat.shape[0] != 4:
            raise ValueError(f"solve_ik: quat must be length 4 (wxyz), got {len(quat)}")

        pos_wrist = np.array(pos_wrist)
        if pos_index is None:
            pos_index = pos_wrist + np.array([0, 0, 0.05])
        if pos_thumb is None:
            pos_thumb = pos_wrist + np.array([0, 0, 0.05])

        pos_index = np.array(pos_index)
        pos_thumb = np.array(pos_thumb)

        q_start = self.data.qpos.copy()
        for j_name, target_val in IK_POSTURE_LOCKS.items():
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id != -1:
                q_start[self.model.jnt_qposadr[j_id]] = target_val

        self.configuration.update(q_start)
        rotation = mink.SO3(quat)

        if posture_target is not None:
            self.tasks[3].set_target(posture_target)
        else:
            self.tasks[3].set_target(self.model.qpos0)

        if posture_cost is not None:
            self.tasks[3].cost = np.array([posture_cost])
        else:
            self.tasks[3].cost = np.array([1e-6])

        self.tasks[0].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_index)
        )
        self.tasks[1].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_thumb)
        )
        self.tasks[2].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_wrist)
        )

        for i in range(500):
            solver = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt=0.15,
                solver="osqp",
                limits=self.limits,
            )

            q_ref = self.configuration.q.copy()
            full_vel = solver.copy()

            for d in range(len(full_vel)):
                if d not in self.auth_dofs:
                    full_vel[d] = 0.0

            mujoco.mj_integratePos(self.model, q_ref, full_vel, 0.05)
            self.configuration.update(q_ref)

            err = self.tasks[0].compute_error(self.configuration)
            if np.linalg.norm(err) < 0.01:
                break

        return self.configuration.q.copy()

    def sync_ctrl_to_qpos(self, q):
        for a_id in range(self.model.nu):
            j_id = self.model.actuator_trnid[a_id, 0]
            q_idx = self.model.jnt_qposadr[j_id]
            self.data.ctrl[a_id] = q[q_idx]

    def get_physics_state(self):
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_pos = np.zeros(3)
        if cube_id != -1:
            cube_pos = self.data.qpos[
                self.model.jnt_qposadr[cube_id] : self.model.jnt_qposadr[cube_id] + 3
            ].copy()

        index_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "R_index_tip_link"
        )
        thumb_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "R_thumb_tip_link"
        )

        hand_pos = np.zeros(3)
        if index_id != -1 and thumb_id != -1:
            p_index = self.data.xpos[index_id]
            p_thumb = self.data.xpos[thumb_id]
            hand_pos = (p_index + p_thumb) / 2.0

        target_dist = float(np.linalg.norm(hand_pos - cube_pos))
        is_grasping = self.current_phase >= 3

        return {
            "cube_z": float(cube_pos[2]),
            "is_grasping": is_grasping,
            "target_dist": target_dist,
        }

    def render_and_record(self, action_32):
        views = {}
        need_depth = self._render_needs_depth()
        for name in self.cam_names:
            self.renderer.update_scene(self.data, camera=name)
            rgb = self.renderer.render()
            depth = None
            if need_depth:
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render().copy()
                self.renderer.disable_depth_rendering()
            self._post_render_hook(name, rgb, depth=depth)
            views[name] = rgb
            self.frame_indices[name] += 1

        self.render_step_idx += 1
        self.rerun_count += 1

    def dispatch_action(
        self, action_32_norm, target_q, n_steps=None, render_freq=None, reset_start=True
    ):
        total_steps = n_steps if n_steps is not None else 200
        rf = render_freq if render_freq is not None else 16

        if reset_start or not hasattr(self, "_last_interp_q"):
            self._last_interp_q = self.data.qpos.copy()

        start_q = self._last_interp_q
        root_target = target_q[self.root_q_idx : self.root_q_idx + 7]

        for step in range(total_steps):
            alpha = (step + 1) / float(total_steps)
            current_target_q = start_q + alpha * (target_q - start_q)
            self.sync_ctrl_to_qpos(current_target_q)
            self.data.qpos[self.root_q_idx : self.root_q_idx + 7] = root_target
            self.data.qvel[:6] = 0.0

            mujoco.mj_step(self.model, self.data)

            if rf > 0 and step % rf == 0:
                self.render_and_record(action_32_norm)

        if rf >= total_steps or rf == 0:
            self.render_and_record(action_32_norm)

        self._last_interp_q = target_q.copy()

    def reset_env(self, lock_posture=False, randomize_cube=True):
        self.current_phase = 0
        self.frame_indices = {cam: 0 for cam in self.cam_names}
        self.render_step_idx = 0
        self.rerun_count = 0
        if randomize_cube:
            rx, ry = np.random.uniform(0.27, 0.63), np.random.uniform(-0.23, 0.23)
        else:
            rx, ry = 0.5, 0
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_q_idx = self.model.jnt_qposadr[cube_id]
        self.data.qpos[cube_q_idx : cube_q_idx + 3] = [rx, ry, 0.82]
        self.data.qpos[cube_q_idx + 3 : cube_q_idx + 7] = [1, 0, 0, 0]
        home_q = self.model.qpos0.copy()
        home_q[cube_q_idx : cube_q_idx + 7] = self.data.qpos[
            cube_q_idx : cube_q_idx + 7
        ].copy()
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1 and self.v_allowed_mask[i] > 0.5:
                name = COMPACT_WIRE_JOINTS[i]
                if name in FROZEN_JOINTS:
                    home_q[self.model.jnt_qposadr[j_id]] = FROZEN_JOINTS[name]
                elif name in IK_POSTURE_LOCKS:
                    target = IK_POSTURE_LOCKS[name]
                    home_q[self.model.jnt_qposadr[j_id]] = (
                        target
                        if lock_posture
                        else target + np.random.uniform(-0.1, 0.1)
                    )
                else:
                    center = (self.wire_max[i] + self.wire_min[i]) / 2.0
                    home_q[self.model.jnt_qposadr[j_id]] = center + np.random.uniform(
                        -0.2, 0.2
                    )
        home_q[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]

        self.data.qpos[:] = home_q.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._last_interp_q = self.data.qpos.copy()
        self.last_target_q = home_q.copy()
        self.render_and_record(None)

    def wild_reset(self):
        print(f"🌀 WILD RANDOMIZING robot pose...")
        self.current_phase = 0

        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_q_idx = self.model.jnt_qposadr[cube_id]
        current_cube_qpos = self.data.qpos[cube_q_idx : cube_q_idx + 7].copy()

        home_q = self.model.qpos0.copy()
        home_q[cube_q_idx : cube_q_idx + 7] = current_cube_qpos

        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1:
                q_idx = self.model.jnt_qposadr[j_id]
                if i < 16:
                    home_q[q_idx] = self.data.qpos[q_idx]
                else:
                    home_q[q_idx] = np.random.uniform(
                        self.wire_min[i], self.wire_max[i]
                    )

        home_q[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]

        self.data.qpos[:] = home_q.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_interp_q = self.data.qpos.copy()
        self.last_target_q = home_q.copy()
        self.render_and_record(None)

    def process_target_32(self, action_32_norm):
        self.active_joints_this_command.clear()
        action_32_rad = self.unscaler.unscale_action(action_32_norm)

        for i, val_norm in enumerate(action_32_norm):
            if (
                not np.isnan(val_norm)
                and self.v_allowed_mask[i] > 0
                and self.protocol_joint_ids[i] != -1
            ):
                self.active_joints_this_command.add(i)
                q_idx = self.model.jnt_qposadr[self.protocol_joint_ids[i]]
                rad = float(action_32_rad[i])
                self.last_target_q[q_idx] = rad

                if i in self.coupling_map:
                    for distal_idx in self.coupling_map[i]:
                        self.last_target_q[distal_idx] = rad

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
