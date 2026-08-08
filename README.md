Here is the comprehensive, production-ready `README.md` for your project repository. It incorporates your updated architectural choices (e.g., removing CLIP/point clouds in favor of DINO, VGGT, and Sobel Edge representations), details Stage 1 and Stage 2 within this updated modality context, and presents a complete breakdown of the 3-phase Stage 3 RL and self-distillation pipeline.

---

# LatentFlow: Latent Dynamics & Multi-Stream Action Flow Matching for Humanoid Manipulation

LatentFlow is a lightweight, generalist Vision-Language-Action (VLA) and Reinforcement Learning framework designed for continuous humanoid robot control (specifically for the **Fourier GR-1** humanoid robot on tabletop manipulation tasks) without relying on hardware-heavy setups or manual physical trajectory generation.

By defining user intent visually and semantically in a static frame—selecting region segments and direction vectors via depth map unprojection—LatentFlow replaces manual teleoperation and physical trial-and-error with an adaptive, simulation-driven continuous latent policy.

---

## Technical Overview & Modality Updates

![System Architecture](assets/architecture_diagram.png)

The model consists of three core components: an encoder for multi-modal feature fusion, a flow matching action planner, and an energy-guided predictor for candidate steering.

* **Encoder**: Processes **5-View Observations** by passing **RGB** visual inputs through parallel perception branches (**DINOv3**, **VGGT**, and **Sobel Filter**) paired with corresponding adapters (**VisAdapter**, **VGGTAdapter**, **EdgeAdapter**), alongside **proprioception** via a **StateAdapter**. Features are aggregated through a **Multi-Stream Transformer** to produce state latent $s_t$.
* **Flow Matcher**: Takes state $s_t$ and target state $s_{target}$ to denoise noisy sample $x_0$ into action trajectory $x_1$, sampling **16 candidate** actions $a_t$.
* **Predictor**: Evaluates predicted next state $s_{t+1}$ against $s_{target}$ via **cosine distance** to compute energy. The computed energy steers actions to produce $a_{steered}$.
* **Execution & Distillation Loop**: 
  * `/step`: Action steering loop ($a_t \rightarrow a_{steered}$).
  * `/calibrate`: Executes $a_{steered}$ directly in simulation.
  * `/distill`: Distills the model to predict the steered action version directly.

1. **Perception Encoders & Modality Adapters**: Raw observations across 5 camera views are processed through frozen foundation backbones:
* **DINO**: Captures high-level semantic layout and patch correspondences.
* **VGGT (Visual Geometry Grounded Transformer)**: Extracts camera tracking and 3D motion fields.
* **Sobel Edge Representation**: Encodes high-frequency structural boundaries.
* Each stream passes through dedicated **MLP Adapters** into a **Multi-Stream Transformer (MST)** to produce a unified $512$-dimensional latent state representation $s_t$.


2. **JEPA Latent Dynamics Predictor**: A deep, multi-layer conditional predictor that models transition dynamics purely in feature space. It takes the current latent state $s_t$ and a proposed joint action array, predicting the future latent state $\hat{s}_{t+1}$. It evaluates physical compatibility and serves as an **Energy-Based Model (EBM)** where the scalar energy $E$ represents distance to the goal:

$$\text{Energy } E = 1.0 - \text{CosineSimilarity}(\hat{s}_{\text{next}}, s_{\text{goal}})$$


3. **Flow-Matching Action Denoiser**: A continuous generative policy head that takes the current state $s_t$, a goal condition $s_{\text{target}}$, and a noisy trajectory. It iteratively denoises the input to generate multi-step continuous joint action candidates over a horizon of 7 ($16 \times 7 \times 58$). To promote exploration without divergence, **steerable stochasticity noise** is injected asynchronously across joint dimensions at each denoising step.

---

## 3-Stage Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Pre-Training                                                  │
│  - Foundation feature extraction (DINO, VGGT, Sobel Edge).             │
│  - Predictor dynamics training over video/demonstration transitions.   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Supervised Fine-Tuning (SFT)                                  │
│  - Train adapters & Action Flow Matcher using offline trajectories. │
│  - Conditional Flow Matching (CFM) & State-Action alignment.            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Online RL Alignment & Self-Distillation                        │
│  - User Annotation Unprojection & IK Positive/Negative Anchors.         │
│  - Epoch Loop: 16 Steps ──► 4 Calibrates ──► 1 Distill.                │
└─────────────────────────────────────────────────────────────────────────┘

```

---

### Stage 1: Pre-Training (Latent Dynamics)

* **Goal**: Ground the JEPA Predictor in physical transition dynamics without requiring real robot joint hardware or explicit action labels.
* **Data**: The dataset mixture contains samples from Droid, CMU Stretch and PointOdyssey.
* **Modality Processing**: Input video sequences are converted into multimodal tokens using frozen **DINO**, **VGGT**, and **Sobel Edge** filters.
* **Mechanism**: The predictor learns to forecast latent representations of future frames given abstract latent action vectors $z_a$. Multi-step autoregressive rollouts and structural regularizers prevent representation collapse.

### Stage 2: Supervised Fine-Tuning (SFT & Action Grounding)

* **Goal**: Train the Action Flow Matcher and modality adapters to map high-level visual/target states to concrete physical joint commands on the Fourier GR-1 robot.
* **Data**: The dataset mixture contains samples from ALOHA, T-REX and Fourier ActionNet.
* **Mechanism**: Using teleoperated demonstration datasets from diverse embodiments, the system optimizes:
* **Conditional Flow Matching (CFM) Loss**: Regresses predicted velocity fields $v_\theta$ to transport noise into valid action trajectories.
* **Contrastive Action-State Alignment (CASA)**: Aligns state transition tokens with continuous motor action embeddings.



---

### Stage 3: Interactive RL Alignment & On-Policy Self-Distillation

Stage 3 aligns the policy to new, user-specified tasks in simulation (e.g., in MuJoCo / Genesis using the Fourier GR-1 humanoid and a target object) using contrastive trajectory feedback, EBM gradient steering, and self-distillation.

#### 1. Setup & Unprojection Phase (Pre-Training Initialization)

1. **User Intent Annotation**: The user selects two segments on a static observation frame:
* **Start Segment**: The effector that moves (e.g., right hand link).
* **Target Segment**: The target object (e.g., a red tabletop cube).
* **Direction Arrow**: Specifies the intended movement vector.


2. **Depth Map 3D Unprojection**: Using the simulation's depth buffer, the 2D segment pixels are unprojected into 3D camera/world space. The system queries the robot's URDF/XML definitions to identify the exact link body names and spatial boundaries.
3. **IK Anchor Generation**: Inverse Kinematics (IK) automatically generates two reference trajectory banks:
* **5 Positive Trajectories ($D^+$)**: Reaching within a close vicinity of the target object 3D extent.
* **8 Negative Trajectories ($D^-$)**: Reaching away from or missing the target object.



---

#### 2. Training Loop Mechanics

Training is structured across **20 Epochs**. Each epoch consists of a precise $16 : 4 : 1$ execution cadence:

$$\text{1 Epoch} = 16 \times \text{\texttt{step}} \;\longrightarrow\; 4 \times \text{\texttt{calibrate}} \;\longrightarrow\; 1 \times \text{\texttt{distill}}$$

```
  Step Call 1..4   ──► Calibrate Call 1 (Trains Predictor)
  Step Call 5..8   ──► Calibrate Call 2 (Trains Predictor)
  Step Call 9..12  ──► Calibrate Call 3 (Trains Predictor)
  Step Call 13..16 ──► Calibrate Call 4 (Trains Predictor)
                                │
                                ▼
                       Distill Call (Trains Flow Matcher)

```

---

#### Phase A: The `step` Endpoint (Action Candidate Exploration & EBM Steering)

1. **Candidate Generation**: The current state $s_t$ and target state $s_{\text{goal}}$ are passed to the Action Flow Matcher. Steerable stochastic noise generates **16 candidate action trajectories**, each of shape $(7, 58)$ (horizon 7, 58 action/joint dimensions).
2. **Predictive Rollout & Energy Scoring**: Each of the 16 candidate trajectories is passed to the JEPA Predictor alongside $s_t$ to predict its resulting future state $\hat{s}_{t+1}$. The normalized cosine distance between $\hat{s}_{t+1}$ and the goal state bank $s_{\text{goal}}$ defines the trajectory's **Energy ($E$)**.
3. **Action Space Steering (No Model Parameter Updates)**:
* Over a loop of 5–8 iterations, the gradient of the energy with respect to the action candidates ($\nabla_a E$) is computed.
* Actions are iteratively adjusted down the energy gradient: $a \leftarrow a - \eta \cdot \nabla_a E$.
* Joints with higher movement activity or higher gradient variance are steered more aggressively while preserving joint limit constraints.


4. **Simulation Execution**: The resulting steered trajectories $(16, 7, 58)$ are executed in the simulation environment to capture real resulting observations and stored in the offline buffer.

---

#### Phase B: The `calibrate` Endpoint (Dynamics Predictor Tuning)

* **Frequency**: Triggered once every 4 `step` calls.
* **Goal**: Update the JEPA Predictor parameters using real simulation rollouts collected during the preceding steps.
* **Mechanism**:
* Encodes $s_t$ and $s_{t+1}$ using current adapters.
* Trains the Predictor to accurately forecast physical transition dynamics $g_\theta(s_t, a) \to \hat{s}_{t+1}$.
* Applies an anti-collapse regularizer (**SIGReg**) to prevent shortcut predictions ($\hat{s}_{t+1} \approx s_t$).



---

#### Phase C: The `distill` Endpoint (Flow Matcher Self-Distillation & Contrastive Sculpting)

* **Frequency**: Triggered once after 16 `step` calls (end of epoch).
* **Goal**: Compress iterative EBM steering directly into the feedforward weights of the Flow Matcher (on-policy self-distillation) and sculpt the contrastive energy landscape.
* **Information Asymmetry**: The steered candidate trajectories possess privileged foresight derived from iterative predictor rollouts. The Flow Matcher (student) is trained to predict these finalized steered trajectories in a single feedforward pass given only the initial observation state $s_t$.
* **Loss Functions**:
1. **Distillation / CFM Loss**: Trains the Flow Matcher to output the steered trajectory directly.
2. **Contrastive Trajectory Loss**: Maximizes margin distance between positive ($D^+$) and negative ($D^-$) trajectory banks in feature space:

$$\mathcal{L}_{\text{contrast}} = \Vert{}a_{\text{pred}} - a_{D^+}\Vert{}^2 + \max(0, M - \Vert{}a_{\text{pred}} - a_{D^-}\Vert{}^2)$$





---

## Directory & File Structure

```
latent-flow/
├── config/
│   └── default_config.yaml         # Architecture hyperparameter definitions
├── models/
│   ├── adapters.py                 # DINO, VGGT, Edge, and Action adapters
│   ├── mst.py                      # Multi-Stream Cross-Attention Transformer
│   ├── jepa_predictor.py           # Energy-Based World Predictor
│   └── action_denoiser.py          # Action Hierarchical DiT Flow Matcher
├── ui/
│   ├── server.py                   # Local WebSocket & FastAPI Simulation Server
│   ├── depth_unprojector.py        # 2D Annotation to 3D World Frame Unprojection
│   ├── ik_trajectory_sampler.py    # Automatic Positive (D+) / Negative (D-) IK Generation
│   └── telemetry.py                # Telemetry & Video Unrolling Utilities
├── colab_server_pkg/
│   ├── feature_extractor.py        # Batched GPU feature extraction (DINO/VGGT/Sobel)
│   ├── stage3_endpoints.py         # Stage 3 step, calibrate, and distill handlers
│   └── image_utils.py              # Base64 image decoding & visualization utilities
├── poc/                            # Self-contained mathematical proof-of-concepts
└── train.py                        # Entry point for Stage 1, Stage 2, and Stage 3

```

---

## Quickstart & Execution Reference

### 1. Environment Setup

```bash
git clone https://github.com/your-username/latent-flow.git
cd latent-flow
pip install -r requirements.txt

```

### 2. Local Simulation Server & Command Center UI

Launch the local FastAPI / WebSocket telemetry server and UI frontend:

```bash
# Start local simulation server (Port 8000)
cd latent-flow/ui
uvicorn server:app --reload --port 8000

```

In a separate terminal, launch the Vite UI dashboard:

```bash
cd latent-flow/ui
npm install && npm run dev
# Open http://localhost:5173 in browser

```

### 3. Training Entrypoints

```bash
# Stage 1: Latent Dynamics Pre-training
python train.py stage=1

# Stage 2: Supervised Fine-Tuning (SFT)
python train.py stage=2

# Stage 3: Interactive RL Alignment (Requires local UI server connection)
python train.py stage=3

```
