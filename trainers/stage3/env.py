import numpy as np
import torch
import sys
import os

# Align paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from simulation_base import GR1MuJoCoBase


class GR1Stage3Env(GR1MuJoCoBase):
    """
    MuJoCo Physics Environment Wrapper for Stage 3 RL.
    Subclasses the clean localized GR1MuJoCoBase to stream true physics states,
    proprioception, and contact grids.
    """

    def __init__(self, action_dim=12, state_dim=24):
        # Initialize the underlying physics model
        super().__init__(restrict_ik=True)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.step_count = 0
        self.max_steps = 100

        # Body indicators
        import mujoco

        self.index_id = self.model.body("R_index_tip_link").id
        self.thumb_id = self.model.body("R_thumb_tip_link").id
        self.cube_id = self.model.body("cube").id

    def action_32_to_qpos(self, action):
        action_32 = action[:32]
        action_rad = self.unscaler.unscale_action(action_32)
        qpos = self.data.qpos.copy()
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1:
                q_idx = self.model.jnt_qposadr[j_id]
                qpos[q_idx] = action_rad[i]
                if i in self.coupling_map:
                    for distal_idx in self.coupling_map[i]:
                        qpos[distal_idx] = action_rad[i]
        return qpos

    def reset(self):
        self.reset_env(lock_posture=True)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        # Extract true proprioceptive joint positions (first 24 dimensions)
        qpos_proprio = torch.tensor(
            self.get_state_32()[: self.state_dim], dtype=torch.float32
        )

        # Get visual frame using center camera
        self.renderer.update_scene(self.data, camera="world_center")
        rgb = self.renderer.render()

        # Flatten image to match Stage 1 visual adapters (384 dimensions)
        vision_feat = torch.tensor(rgb[:16, :8, :].astype(np.float32).flatten()[:384])
        if len(vision_feat) < 384:
            vision_feat = torch.cat([vision_feat, torch.zeros(384 - len(vision_feat))])

        # Mock PointNeXt (384) & VGGT (768) embeddings for compatibility
        pointnext = torch.randn(384)
        vggt = torch.randn(768)

        # Calculate true tactile grid from body coordinates
        index_pos = self.data.xpos[self.index_id]
        thumb_pos = self.data.xpos[self.thumb_id]
        cube_pos = self.data.xpos[self.cube_id]

        d_index = float(np.linalg.norm(index_pos - cube_pos))
        d_thumb = float(np.linalg.norm(thumb_pos - cube_pos))

        touch_index = max(0.0, 1.0 - (d_index / 0.04)) if d_index < 0.04 else 0.0
        touch_thumb = max(0.0, 1.0 - (d_thumb / 0.04)) if d_thumb < 0.04 else 0.0

        # 4x4 matrix for tactile adapter compatibility
        tactile_grid = torch.zeros(4, 4)
        tactile_grid[0, 0] = touch_index
        tactile_grid[1, 1] = touch_thumb

        return {
            "vision": vision_feat.unsqueeze(0),
            "pointnext": pointnext.unsqueeze(0),
            "vggt": vggt.unsqueeze(0),
            "tactile": tactile_grid.unsqueeze(0),
            "proprioception": qpos_proprio.unsqueeze(0),
            "text": torch.randn(1, 1, 512),
        }

    def step(self, action, perturb_force=None):
        self.step_count += 1

        # Apply action to simulator control parameters
        effective_action = action.clone()
        if perturb_force is not None:
            effective_action += perturb_force

        # Apply target joint posture from action (Standardized conversion)
        qpos_target = self.action_32_to_qpos(effective_action.squeeze(0).cpu().numpy())
        self.sync_ctrl_to_qpos(qpos_target)

        # Stabilize humanoid torso
        self.data.qpos[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]
        self.data.qpos[self.root_q_idx + 3 : self.root_q_idx + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:6] = 0.0

        # Step physics engine
        import mujoco

        mujoco.mj_step(self.model, self.data)

        # Compute dense reward based on fingertip-to-cube distance
        physics = self.get_physics_state()
        target_dist = physics["target_dist"]
        reward = -target_dist

        # Check collision constraints
        done = self.step_count >= self.max_steps
        tactile_spike = float(target_dist < 0.03)

        obs = self._get_obs()
        return (
            obs,
            reward,
            done,
            {"tactile_spike": tactile_spike, "target_dist": target_dist},
        )
