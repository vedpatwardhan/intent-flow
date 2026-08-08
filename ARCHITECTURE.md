# IntentFlow Architecture & Technical Specification

This document details the deep mathematical, architectural, and algorithmic foundations of **IntentFlow**.

---

## 1. System Overview

IntentFlow translates high-level 2D viewport intent annotations (drawn via static camera frames) into 3D continuous motor actions for tabletop humanoid robot manipulation (**Fourier GR-1**).

<p align="center">
  <img src="assets/diagram.jpg" alt="System Architecture" width="600" />
</p>

---

## 2. Perception & Multi-Stream Transformer (MST)

Raw observations across 5 camera views (`world_center`, `world_top`, `world_left`, `world_right`, `world_wrist`) are processed through parallel frozen vision backbones paired with dedicated MLP adapters:

- **DINOv3 (`VisAdapter`)**: Extracts patch-level semantic correspondences and global spatial layout ($d_{\text{in}} = 384$).
- **VGGT (`VGGTAdapter`)**: Visual Geometry Grounded Transformer extracting 3D motion fields ($d_{\text{in}} = 384$).
- **Sobel Edge (`EdgeAdapter`)**: High-frequency structural boundary heatmap features ($d_{\text{in}} = 384$).
- **Proprioception (`StateAdapter`)**: 32-dimensional robot joint configuration tokens ($d_{\text{in}} = 32$).

Tokens are aggregated across modalities through a **Multi-Stream Transformer (MST)** to produce a unified 512-dimensional state latent vector:

$$s_t = \text{MST}(\mathbf{z}_{\text{vis}}, \mathbf{z}_{\text{vggt}}, \mathbf{z}_{\text{edge}}, \mathbf{z}_{\text{proprio}}) \in \mathbb{R}^{512}$$

---

## 3. JEPA Latent Dynamics Predictor (Energy-Based Model)

The **JEPA Predictor** ($g_\theta$) models physical transition dynamics purely in feature space:

$$g_\theta(s_t, a_t) \longrightarrow \hat{s}_{t+1} \in \mathbb{R}^{512}$$

The scalar **Energy ($E$)** of a predicted state relative to a target goal state $s_{\text{goal}}$ is computed via normalized cosine distance:

$$\text{Energy } E = 1.0 - \frac{\hat{s}_{t+1} \cdot s_{\text{goal}}}{\|\hat{s}_{t+1}\|_2 \, \|s_{\text{goal}}\|_2}$$

During action candidate evaluation (`/step`), energy gradients $\nabla_a E$ iteratively steer 16 flow-matched action candidates down the energy landscape.

---

## 4. Action Flow Matcher Denoiser

The **Action Flow Matcher** ($\mathbf{v}_\theta$) is a continuous generative policy head that predicts joint velocity fields:

$$\mathbf{v}_\theta(x_t, t, s_t, s_{\text{target}}) \in \mathbb{R}^{7 \times 58}$$

### Loss Formulations

1. **Conditional Flow Matching (CFM) Loss**:
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \left[ \left\| \mathbf{v}_\theta(x_t, t, s_t, s_{\text{target}}) - (x_1 - x_0) \right\|_2^2 \right]$$

2. **Contrastive Trajectory Sculpting Loss**:
$$\mathcal{L}_{\text{contrast}} = \left\| a_{\text{pred}} - a_{D^+} \right\|_2^2 + \max\left(0, \, M - \left\| a_{\text{pred}} - a_{D^-} \right\|_2^2 \right)$$

where $D^+$ (5 positive trajectories) and $D^-$ (8 negative trajectories) are generated via Inverse Kinematics (IK) from unprojected viewport annotations.

---

## 5. 3-Stage Training Pipeline & Cadence

```
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Encoder and Predictor Training                │
│  - Foundation feature extraction (DINO, VGGT, Sobel).  │
│  - Predictor dynamics over demonstration transitions.  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Action Denoiser Training                      │
│  - Train adapters & Action Flow Matcher offline.       │
│  - Conditional Flow Matching & State-Action alignment. │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Online RL Alignment & Self-Distillation       │
│  - Unprojection & IK Positive/Negative Anchors.        │
│  - Epoch Loop: 16 Steps ──► 4 Calibrates ──► 1 Distill.│
└────────────────────────────────────────────────────────┘
```

### Stage 3 Execution Cadence (16 : 4 : 1)

Each epoch consists of:
- **16 $\times$ `/step`**: Generates 16 action candidates per step, steers candidates via EBM predictor gradients ($\nabla_a E$), and executes rollout on MuJoCo / Genesis simulation.
- **4 $\times$ `/calibrate`**: Triggered every 4 steps to update JEPA Predictor weights on real physical transition tuples using SIGReg anti-collapse regularizer.
- **1 $\times$ `/distill`**: Triggered at epoch end to distill EBM-steered trajectories into single-pass feedforward weights of the Action Flow Matcher and compute latent energy landscape analytics.

---

## 6. Pre-Training & SFT Datasets

To train IntentFlow on standard GPU environments, we employ a specific dataset mix across Stage 1 and Stage 2:

### A. Stage 1: Encoder and Predictor Training (60:30:10 Mix)

1. **60% Tabletop Robot Manipulation (`lerobot/droid`)**:
   - grounds the dynamics predictor in robot joint space constraints and physical tabletop transitions.
2. **30% Human Tabletop Hand-Object Interaction (`lerobot/cmu_stretch` & Ego4D)**:
   - Teaches hand-object affordances, tool-use semantics, and visual contact priors from first-person actions.
3. **10% 3D Visual Geometry & Tracking (`PointOdyssey`)**:
   - Anchors predictor transitions in 3D topological tracking and geometric camera transformations.

#### Modality Processing Rules across Stage 1 Streams:

| Modality / Stream | Stream A: Droid (Robot) | Stream B: CMU Stretch (Human) | Stream C: PointOdyssey (Geometry) |
| :--- | :--- | :--- | :--- |
| **2D Video Frames (RGB)** | **Present** (Franka cameras) | **Present** (Egocentric camera) | **Present** (CG camera render) |
| **Text Instruction** | **Present** (Task prompt) | **Present** (Task prompt) | **Present** (Task prompt) |
| **Actions (Robot joints)** | **Present** (Franka joints) | **Missing** (Zero-masked) | **Missing** (Zero-masked) |
| **Proprioception (Robot state)** | **Present** (Franka joint angles) | **Missing** (Zero-masked) | **Missing** (Zero-masked) |
| **Tactile Grid (Touch)** | **Missing** (Zero-masked) | **Missing** (Zero-masked) | **Missing** (Zero-masked) |

---

### B. Stage 2: Action Denoiser Training (70:30 Mix)

1. **70% Bimanual ALOHA Tabletop & Fourier ActionNet (`lerobot/aloha_static_coffee`, `lerobot/fourier_actionnet`)**:
   - Provides paired camera frames and 58-dim humanoid joint torque/velocity commands.
   - Directly grounds the control policy in Fourier GR-1 humanoid joint limits and morphology.
2. **30% T-REX Contact-Rich Tactile Dataset (`zekaiwang/trex_dataset`)**:
   - Provides high-frequency $4 \times 4$ fingertip contact pressure vectors to train contact-rich tactile alignment.
