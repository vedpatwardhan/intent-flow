# Latent Flow

---

### High-Level Component Architecture Diagram

The diagram below highlights the main architectural components and their core relationships, detailing the path from raw observations and user prompts through the parallel perception pathways (including SAM masking filters and VGGT tracking) into the trainable adapters, MSAT cross-attention fusion, and core model loop.

```
                               [ USER INPUT ]
       ┌──────────────┬──────────────┬──────────────┬──────────────┐
       │ (Text Inst)  │ (2D Frames)  │ (3D Cloud)   │ (Tactile/Prop)
       ▼              ▼              │              ▼              ▼
 ┌───────────┐  ┌───────────┐        │ (Click)      │        ┌───────────┐
 │ CLIP Text │  ├─► DINOv3  │        ▼              │        │ Sensor IO │
 └─────┬─────┘  │           │   ┌───────────┐       │        └─────┬─────┘
       │        ├─► VGGT    ├──►│ SAM & KLT │       │              │
       │        └─────┬─────┘   └─────┬─────┘       │              │
       │              │         ┌─────▼─────┐       │              │
       │              │         │ PointNeXt │◄──────┘              │
       │              │         └─────┬─────┘                      │
       ▼              ▼               ▼                            ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ 2. TRAINABLE MLP ADAPTERS (Text, DINO, VGGT, PointNeXt, Tactile) │
 └───────────────────────────────┬──────────────────────────────────┘
                                 ▼ (Shared 512-dim tokens)
 ┌──────────────────────────────────────────────────────────────────┐
 │ 3. MSAT CROSS-ATTENTION FUSION LAYER                             │
 └───────────────────────────────┬──────────────────────────────────┘
                                 ▼ (Unified State s_t)
 ┌──────────────────────────────────────────────────────────────────┐
 │ 4. CORE LATENT MODELS (Learnable Dynamics & Policy)              │
 │    ┌────────────────────────────────────────────────────────┐    │
 │    │              JEPA World Predictor (EBM)                │    │
 │    └───────────────────────────▲───┬────────────────────────┘    │
 │                                │   │ (DAWN Reciprocity Loop)     │
 │                   [ Action Adapter (f_action) ]                  │
 │                                ▲   │                             │
 │    ┌───────────────────────────┴───▼────────────────────────┐    │
 │    │        Flow-Matching Action Denoiser (CLAP-RF)         │    │
 │    └───────────────────────────▲───┬────────────────────────┘    │
 └────────────────────────────────║───║──────────────────────────────┘
                                  ║   ║ (Proposed Actions)
                                  ║   ▼
 ┌────────────────────────────────║───║──────────────────────────────┐
 │ 5. SAFETY & FEASIBILITY REGULATORS                               │
 │    pycapacity Workspace Polytope ──► EBT Safeguards (Langevin)   │
 └──────────────────────────────────────────────────────────────────┘
```��──────────────────────────┘    │
 │                                   │   │ (DAWN Reciprocity Loop)          │
 │                      [ Action Adapter (f_action) ]                       │
 │                                   ▲   │                                  │
 │    ┌──────────────────────────────┴───▼─────────────────────────────┐    │
 │    │           Flow-Matching Action Denoiser (CLAP-RF)              │    │
 │    └──────────────────────────────▲───┬─────────────────────────────┘    │
 └───────────────────────────────────║───║──────────────────────────────────┘
                                     ║   ║
                   (Refined Actions) ║   ║ (Proposed Actions)
                                     ║   ▼
 ┌───────────────────────────────────╩───╩──────────────────────────────────┐
 │ 4. SAFETY & FEASIBILITY REGULATORS                                       │
 │    pycapacity Workspace Polytope ──► EBT Safeguards (Langevin / MCMC)    │
 └──────────────────────────────────────────────────────────────────────────┘

 ┌────────────────────────────────────────┐ ┌───────────────────────────────┐
 │ 5. HYBRID MEMORY TRIAD                 │ │ 6. MODULAR SKILL CHAINING     │
 │  - Short-Term Context (Sliding Window) │ │  - d-OPSD Self-Distillation   │
 │  - Event Boundary (Full Latents)       │ │  - RATs Task Proposer         │
 │  - Long-Range Gist (HyDRA Tokens)      │ │  - PSN Skill Library          │
 └───────────────────┬────────────────────┘ └──────────────┬────────────────┘
                     │ (Prior Context)                     │ (Skill Vectors)
                     └─────────────────────┬───────────────┘
                                           ▼
                           [ Flow-Matching Denoiser ]
```

---

## Encoder Streams (A1 - A5) Details

To build a generalist policy capable of zero-shot transfer, we leverage heterogeneous multimodal encoders that translate raw environment observations into compact latent representations:

*   **A1: Language (CLIP Text)**: Focuses on task and semantic coaching. By embedding natural language descriptions (e.g., *"pinch the cube's corners"*) and general semantic categories, it conditions the flow-matching denoiser to align target behavior with the task goal.
*   **A2: DINOv3 Full-Frame Vision**: Runs on *every single incoming image frame* (multi-view raw inputs) instead of static waypoints. DINOv3 serves as a self-supervised visual backbone that extracts dense semantic correspondences and spatial layouts across time. This acts as a continuous state-tracking reference for the JEPA world predictor.
*   **A3: PointNeXt Egocentric Vision**: Focuses on millimeter-level 3D spatial alignment. During live execution, PointNeXt ingests raw 3D coordinate point sets $(X, Y, Z)$ directly from the robot's depth cameras. During 2D-only pre-training (Stage 1), PointNeXt is fed by running **EgoFlow point tracking**: we estimate depth maps (Depth Anything V2) and segmentation masks (SAM 2) only on the first frame ($t=0$) to select interest keypoints, and propagate these coordinates frame-by-frame instantly using Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) to construct the 3D point cloud sequence.
*   **A4: VGGT Visual Geometry**: Infers dense 3D scene layouts, camera parameters (intrinsic/extrinsic matrices $R, T$), and 3D point tracks across frame sequences. It provides camera trajectories and tracks key coordinates $(X, Y, Z)$ over time, giving the model a temporal motion tracking prior.
*   **A5: Tactile & Proprioceptive Streams**: Captures contact-rich physical feedback (finger forces, joint positions, velocities, and torques). This bridges the gap between vision and physical execution.

---

## Tactile Stream Architecture & MuJoCo Simulation

### 1. Tactile Stream Architecture
The tactile stream is designed to process spatial force feedback during contact-rich tasks (like the pinch and lift phases):
*   **Input Representation**: The tactile input is formatted as a spatial matrix representing contact pressure points across the inner fingertip pads (e.g., an $N \times M$ matrix where $N, M$ represent the sensor grid density) [retrieved at runtime via MuJoCo's `sensordata` array or by filtering raw `data.contact` collision structures].
*   **Encoder Module**: A shallow Convolutional Neural Network (CNN) or multi-layer perceptron (MLP) acts as the Tactile Encoder, compressing the 2D spatial grid into a dense 1D tactile embedding.
*   **Cross-Attention Integration**: This tactile embedding is concatenated with joint proprioception tokens (angles, velocities, torques) and passed to the Multi-Stream Action Transformer (MSAT) via joint cross-attention alongside the visual tokens.

### 2. Simulating Tactile Feedback in MuJoCo
In MuJoCo, tactile sensors are simulated using a grid of `<touch>` sensors placed on the inner fingertips of the humanoid hands.

*   **XML Definition**:
    We define individual touch site nodes and register them in the `<sensor>` block:
    ```xml
    <!-- Define touch sites on the right fingertip pad -->
    <body name="right_fingertip">
        <site name="touch_pad_0_0" pos="0.005 0.010 0" size="0.002"/>
        <site name="touch_pad_0_1" pos="0.005 0.010 0.004" size="0.002"/>
        <site name="touch_pad_1_0" pos="-0.005 0.010 0" size="0.002"/>
        <site name="touch_pad_1_1" pos="-0.005 0.010 0.004" size="0.002"/>
    </body>

    <!-- Map touch sites to sensors -->
    <sensor>
        <touch name="tactile_0_0" site="touch_pad_0_0"/>
        <touch name="tactile_0_1" site="touch_pad_0_1"/>
        <touch name="tactile_1_0" site="touch_pad_1_0"/>
        <touch name="tactile_1_1" site="touch_pad_1_1"/>
    </sensor>
    ```
*   **Runtime Retrieval**:
    During simulation steps, we query the MuJoCo data structure via Python (`mujoco.MjData`):
    ```python
    # Fetch sensor readings from mujoco data
    sensor_data = sim.data.sensordata
    
    # Extract the registered touch sensor readings
    tactile_matrix = np.array([
        [sim.data.sensor("tactile_0_0").value[0], sim.data.sensor("tactile_0_1").value[0]],
        [sim.data.sensor("tactile_1_0").value[0], sim.data.sensor("tactile_1_1").value[0]]
    ])
    ```
    This spatial matrix is normalized and passed directly into the Tactile Encoder at every environment step.

---

## JEPA as an Energy-Based Model (EBM): Alignments, Deviations, and Frozen Encoders

Evaluating the Joint-Embedding Predictive Architecture (JEPA) through Yann LeCun’s foundational Energy-Based Model (EBM) framework reveals deep mathematical alignments, structural deviations, and how the architecture behaves when visual encoders are frozen.

### 1. Alignments: Why JEPA is an Energy-Based Model
An EBM models the dependencies between variables by assigning a scalar value (energy) to each configuration of variables. In JEPA, the variable configuration is $(x, a, y)$, where $x$ is the context observation, $a$ is the action, and $y$ is the future target observation. 
*   **Energy as Latent Prediction Error**: JEPA maps inputs to latent states ($s_x = f_\phi(x)$ and $s_y = f_\phi(y)$) and uses a predictor network $g_\theta$ to estimate the future latent: $\hat{s}_y = g_\theta(s_x, a)$. The energy function $E(x, a, y)$ is defined directly as the prediction distance in the feature space:
    $$E(x, a, y) = D(g_\theta(s_x, a), s_y) = \lVert g_\theta(s_x, a) - s_y \rVert^2$$
    Low prediction error represents high compatibility (low energy), whereas high prediction error represents physically impossible or anomalous transitions (high energy).
*   **Non-Generative Evaluation**: Like standard EBMs, JEPA discards the pixel-reconstruction decoder. It acts purely as a compatibility evaluator, checking if the latent trajectory matches the physical reality of the environment.
*   **MPC Planning via Gradient Minimization**: Because the energy function is continuous and differentiable in the action space, an agent can compute the gradient of the energy with respect to actions ($\nabla_a E$). This allows Model Predictive Control (MPC) and Langevin Dynamics to iteratively refine action proposals by sliding them down the energy gradient to find optimal, low-energy trajectories.

### 2. Deviations: How JEPA Differs from Classical EBMs
JEPA deviates from traditional EBM formulations in its approach to asymmetry and collapse prevention:
*   **Asymmetric Dual-Pathways**: Standard EBMs use a single symmetric network to compute the compatibility of two states. JEPA uses asymmetric encoders (often updating the target encoder via Exponential Moving Average (EMA) and stop-gradients) to provide a moving target, preventing representations from locking into static, degenerate states.
*   **Anti-Collapse Regularization vs. Partition Normalization**: Traditional EBMs prevent collapse (where the energy function becomes flat and zero everywhere) by either running contrastive training to raise the energy of negative samples (using a partition function or contrastive divergence) or using mathematical regularizers. JEPA avoids contrastive negative sampling during pre-training. Instead, it relies on architectural regularization to maximize information capacity:
    *   **VICReg**: Constrains the variance and covariance of the latents to ensure they remain informative and decorrelated.
    *   **SIGReg (Sketched-Isotropic-Gaussian Regularizer)**: Used in **LeWorldModel (LeWM)**, it projects high-dimensional latent variables onto random 1D directions and forces them to match an isotropic Gaussian distribution, mathematically guaranteeing collapse prevention without requiring stop-gradients or EMA.

### 3. Relevance of EBM Properties Under Frozen Pre-trained Encoders
In our architecture, we ignore visual encoder training and use **frozen, pre-trained encoders** (DINOv3, CLIP, PointNeXt, VGGT). Under this configuration, the dynamics behave as a **Conditional or Predictive EBM**:
*   **Elimination of Representation Collapse**: Because the encoders $f_\phi$ are frozen, the latent representations $s_x$ and $s_y$ are static and cannot collapse to a constant vector. This renders encoder-level anti-collapse losses (like VICReg or SIGReg) unnecessary.
*   **The Threat of Predictor Collapse**: While the representation space cannot collapse, the predictor $g_\theta(s_x, a)$ can still suffer from predictor collapse (or action-ignorance), where it learns to predict a static future state ($\hat{s}_y \approx s_x$) regardless of the action $a$. We prevent this via:
    1. **Multi-Step Rollouts**: We train the predictor to roll out over a temporal horizon of $H = 8$ to $12$ steps auto-regressively: $\hat{s}_{t+k} = g_\theta(\hat{s}_{t+k-1}, a_{t+k-1})$. The loss is accumulated over the horizon: $\mathcal{L}_{\text{dynamics}} = \sum_{k=1}^H \gamma^k \| \hat{s}_{t+k} - s_{t+k} \|_2^2$. This prevents copy-paste collapse.
    2. **6-Layer Conditional Transformer Predictor (AdaLN-Zero Modulation)**: Rather than using a simple single-layer MLP concatenation, the dynamics predictor $g_\theta$ utilizes a **6-layer Conditional Transformer** with **AdaLN-zero conditioning blocks** (matching the model capacity of the `le-probe` project). The state tokens $s_t$ are modulated at each block layer using scale/shift/gate parameters calculated dynamically from the action embeddings $z_t$. This functions as a deep, multi-layer generalized gating mechanism that forces mathematical dependency on actions, avoiding predictor collapse while scaling temporal sequence capacity.
*   **Tracking Predictor Collapse at Training Time**:
    We monitor three zero-overhead diagnostic metrics to detect collapse during training:
    1. **"No-Op" Loss Ratio**: $\text{Ratio}_{\text{collapse}} = \frac{\mathbb{E}[ \| g_\theta(s_t, a_t) - s_{t+1} \|^2 ]}{\mathbb{E}[ \| s_t - s_{t+1} \|^2 ]}$. A healthy ratio is significantly below $1.0$. A ratio approaching $1.0$ indicates collapse to copy-paste.
    2. **Action Sensitivity Variance**: $\text{Var}(\hat{s}) = \frac{1}{B} \sum_{i=1}^B \| g_\theta(s_t^{(i)}, a_t^{(i)}) - \bar{\hat{s}} \|_2^2$. If variance drops to zero, the predictor is outputting a constant vector.
    3. **Action Perturbation Drift**: $\Delta_{\text{action}} = \mathbb{E}[ \| g_\theta(s_t, a_t) - g_\theta(s_t, a_{\text{rand}}) \|_2^2 ]$. If this drift drops to zero, the network is ignoring the action input channels.
*   **Stable Energy Landscapes**: In traditional EBMs, training encoders and predictors simultaneously is highly unstable. Freezing the encoders anchors the latent space to fixed semantic (CLIP) and visual-spatial (DINOv3, PointNeXt, VGGT) coordinate systems. The predictor $g_\theta$ is trained purely to minimize the transition energy over this fixed landscape.
*   **Highly Relevant for Trajectory Planning**: The energy function $E(x, a, y) = \lVert g_\theta(s_x, a) - s_y \rVert^2$ remains fully active and differentiable. Since the latent space is stabilized, computing the gradient $\nabla_a E$ for MPC path planning or Langevin trajectory refinement becomes highly reliable, avoiding the "latent detachment" (where the latent space drifts from physical reality) common in end-to-end trained joint-embedding architectures.

### 4. Parameterized EBT-Policy vs. Calculated JEPA-EBM
To distinguish our calculated transition energy from direct parameterized energy (such as the EBT-Policy in arXiv:2510.27545), we evaluate their structural divergence:

| Metric | EBT-Policy (Direct Parameterization) | JEPA-EBM (Calculated Distance) |
| :--- | :--- | :--- |
| **Energy Source** | **Direct Model Output**: A Transformer network directly outputs a single scalar energy score: $E_\theta(s, a)$. | **Calculated Metric**: Computed as the $L_2$ prediction error in latent space: $\lVert g_\theta(s_x, a) - s_y \rVert^2$. |
| **Target Reference** | None. Learns implicitly from regularized Behavior Cloning losses. | **Latent Target**: Uses the future target state representation $s_y$ from the target encoder. |
| **Inference/Planning** | Optimizes $E_\theta(s, a)$ directly w.r.t actions using gradient descent. | Minimizes squared distance $D(\hat{s}_y, s_y)$ w.r.t actions using MPC (CEM/Langevin). |

### 5. Paradigm Shift: Dynamics-Based Pre-Training vs. Contrastive Post-Training
To maximize both pre-training sample efficiency and post-training policy robustness, the training pipeline shifts how it stabilizes the predictor and sculpts the energy landscape between Stage 1 and Stage 3:

#### A. Stage 1 (Pre-training): Dynamics-Based Predictor Grounding
*   **The Goal**: Establish a stable, multi-modal transition dynamics model (JEPA Predictor) over the frozen coordinate systems of DINOv3, CLIP, PointNeXt, and VGGT.
*   **Why Non-Contrastive?**: We do not yet have access to task success boundaries or balanced negative interaction samples. Instead of contrastive training, Stage 1 focuses purely on predicting physical transition trajectories, utilizing **multi-step rollouts** and **bilinear gating** to ensure the predictor is physically grounded and cannot collapse (ignore actions).

#### B. Stage 3 (Post-training/RL): Memory-Driven Contrastive Energy Sculpting
*   **The Goal**: Align the policy to maximize task success and learn recovery behaviors from failures.
*   **Why Contrastive is Safe Here**: The encoders are now fully frozen. The latent representations are anchored, meaning there is **zero risk of representation collapse**.
*   **Contrastive Sculpting via Memory**: As the robot explores the environment, it logs successful grasp trajectories (positives) and failed/near-miss trajectories (negatives, e.g., the 500 catastrophic failures) into the memory triad. We apply a contrastive loss to the EBM predictor $g_\theta$:
    *   **Minimize prediction error (energy)** for positive/expert transitions.
    *   **Maximize prediction error (energy)** ($E \to \infty$) for negative/failure transitions.
    This carves deep, low-energy "valleys" around successful paths and builds high-energy "walls" around failure configurations, forcing the policy to naturally remain within the stable physical manifold.

---

## Decentralized VLA Architecture & Modality Alignment

To achieve low-latency continuous control on the Fourier GR-1 without the VRAM footprint of a full VLA, we discard the heavy high-level reasoning and pre-training modules of the original CLAP framework (arXiv:2601.04061) and deploy only its low-level control component (CLAP-RF). However, we retain CLAP's core advantage of contrastive visual-proprioceptive alignment by utilizing the MLP adapters.

### 1. Architectural Mapping: Picked vs. Discarded Components

The table below outlines how our lightweight architecture adapts and retains CLAP's primary features while discarding its heavy parameter footprints:

| CLAP Original Component | What it does in CLAP | What we do with it | Why? (The Rationale) |
| :--- | :--- | :--- | :--- |
| **Act-VAE / VD-VAE** | Quantizes continuous actions/videos into a discrete token codebook. | **Discard the VAEs**. We retain the *concept* of a unified latent action space but represent it continuously using trainable MLP Adapters. | Discretizing actions via VAE is computationally heavy to train and limits high-frequency continuous control precision. |
| **CLAP-NTP (4B VLM)** | Takes text and vision tokens to autoregressively predict discrete action tokens. | **Discard the VLM**. We replace it with frozen CLIP Text conditioning and programmatic skill sequencing via the **Memory Triad**. | Autoregressive generation from a 4B VLM is too slow for 10Hz+ real-time control and consumes massive VRAM. |
| **CLAP-RF** | A Rectified Flow policy that refines coarse VLM tokens into continuous robot controls. | **Retain the CLAP-RF policy head** as our standalone continuous Action Denoiser. | Flow-matching generates extremely smooth continuous trajectories and handles high-frequency joint control efficiently. |
| **Contrastive Transition Matching** | A contrastive loss aligning visual video frames with robot action features. | **Retain this concept as our CASA Loss** (InfoNCE) applied directly to the MLP Adapters. | It solves "visual entanglement," forcing the visual adapters to ignore background clutter and extract only action-relevant features. |

### 2. Modality Alignment: MLP Adapters and Cross-Attention Fusion
To capture CLAP's core benefit of cross-modal alignment without the VLM parameters, we implement a two-stage alignment and fusion pipeline:
* **Step 1: MLP Projection & Contrastive Alignment (CASA)**:
  * Each frozen perception stream (CLIP Text, DINOv3, PointNeXt, VGGT, Tactile/Proprioceptive) is projected by its own trainable MLP adapter into a unified $d$-dimensional space (e.g., $d=512$).
  * **Stage Transitions**: During Stage 1 pre-training, no contrastive alignment is performed; the adapters are updated purely via the predictive transition loss to capture physical motion. Contrastive Action-State Alignment (CASA) is introduced in **Stage 2 SFT** to ground state representations. We encode robot action trajectories $a_t$ using a simple MLP action encoder $h_\psi(a_t) \to z_a$. The adapters project state transitions into a visual-action latent space $f_{\theta_v}(s_{t:t+1}) \to z_s$.
  * We optimize an InfoNCE loss to maximize the cosine similarity of matching pairs $(z_s, z_a)$ while minimizing it for mismatched pairs in the batch. This forces the adapters to extract only action-relevant features, resolving "visual entanglement."
* **Step 2: Attention-Based Fusion & Modality Parity (MSAT)**:
  * Once the modality tokens are dimensionally matched and contrastively aligned, they are concatenated and fed into the **Multi-Stream Action Transformer (MSAT)**.
  * **Visual Parity Breaking**: In Stage 1, all inputs maintain structural visual parity (DINOv3 full frames and KLT depth track regions are geometrically aligned). In **Stage 3 RL & Downstream execution**, we break this parity: the user's camera click isolates only the target object point cloud (PointNeXt), while DINOv3 continues to process the entire visual scene. MSAT's cross-attention layers learn to route focal attention to the PointNeXt object tokens, ignoring background clutter.
  * **Tactile Masking Evolution**: Since pre-training datasets lack touch skins, the tactile stream is **100% zero-masked during Stage 1**. The adapter learns to treat tactile coordinates as flat zeros. In **Stage 3**, when executing contact-rich tasks in the simulator, tactile grids become active, allowing MSAT to dynamically weigh visual target tokens against finger contact pressures.

### 3. Action Vector Space: Latent Action Encoder & Action Adapter
To pre-train our world dynamics model on datasets containing only video transitions (human or robot videos without joint values) and then map them to physical robot commands, we use a dual-module action mapping setup:

* **Latent Action Encoder ($h_\psi$, Stage 1 Only)**:
  * During dynamics pre-training, we do not have joint torques ($a_t$). We feed the current state $s_t$ and the future state $s_{t+1}$ into a shallow MLP: $z^{\text{latent\_action}}_t = h_\psi(s_t, s_{t+1})$. 
  * **OlafWorld Regularization**: To ensure this latent space is continuous, smooth, and behaves as a well-defined prior for skill composition, we apply a **Variational Information Bottleneck (VIB)** constraint. We output mean and variance vectors to compute a KL-divergence loss against a standard Gaussian prior: $\mathcal{D}_{\text{KL}}(q_\psi(z|s_t, s_{t+1}) \parallel \mathcal{N}(0, I))$.
  * The predictor learns to forecast using this latent action: $\hat{s}_{t+1} = g_\theta(s_t, z^{\text{latent\_action}}_t)$.
  * At Stage 2, this module is completely discarded.
* **Action Adapter ($f_{\text{action}}$, Stage 2 Onwards)**:
  * When we fine-tune on joint command datasets, we feed the physical action $a_t$ (torques/velocities) into a trainable Action Adapter MLP to project it into the latent action dimension: $z^{\text{latent\_action}}_t = f_{\text{action}}(a_t)$.
  * The predictor evaluates transitions using this projected joint representation: $\hat{s}_{t+1} = g_\theta(s_t, f_{\text{action}}(a_t))$.

#### Rationale for the Shallow MLP Architecture
We utilize small, 2-to-3 layer MLPs for $h_\psi$ and $f_{\text{action}}$ because:
1. **Inputs are Pre-Compressed**: The modules do not process raw pixel arrays. They operate on highly compressed $512$-dim semantic representations produced by DINOv3, PointNeXt, and VGGT, which reduces the complexity to simple vector projection.
2. **The Anti-Leakage Bottleneck**: To prevent the model from "cheating" during Stage 1 pre-training, the Latent Action Encoder ($h_\psi$) must be constrained. A deep, high-capacity network would easily memorize the future state $s_{t+1}$ and pass it directly to the predictor, bypassing the dynamics. A shallow MLP acts as a low-dimensional bottleneck ($16$-dim or $32$-dim) that can only convey the abstract motion delta (direction and scale), forcing the predictor to model real physical transitions.

### 4. Pre-Training Prior & Data Feasibility
While we do not train the CLAP encoders from scratch, we leverage the weights and representations pre-trained on:
* **Ego4D (Human Video)**: Provides the visual-physics priors (hand-object boundaries and grasp semantics) embedded in the pre-trained DINOv3 backbone.
* **Astribot S1 & AgiBot (Robot Demos)**: Establishes a highly transferable initialization for dual-arm/humanoid tabletop manipulation inside the pre-trained CLAP-RF weights.

### 5. Compute Budget Feasibility for Independent Researchers
By utilizing the frozen-encoder + adapter strategy and restricting our learnable parameter set to the standalone CLAP-RF policy head:
* **Active Learnable Parameters**: ~35M parameters (trainable MLP adapters + CLAP-RF head).
* **Colab Footprint**: Fine-tuning the adapters and the CLAP-RF head on your datasets.bot teleoperated dataset takes **4 to 8 hours on a single L4 GPU** (~5-10 Colab credits), preserving your remaining budget for RL adaptations.

---

## Compensating for Scale: Skill Chaining & Memory Triad for Long-Horizon Control

Monolithic Reinforcement Learning (training a single neural network to map high-level language directly to motor torques over long trajectories) is computationally intractable on a limited budget. It suffers from the **Credit Assignment Bottleneck** (sparse rewards over long horizons) and **Gradient Interference** (where learning the lift phase overwrites the reach phase).

We bypass these limitations through **Programmatic Skill Chaining**, utilizing the **Hybrid Memory Triad** to track temporal state transitions instead of relying on brute-force policy gradients.

### 1. Training Modular Atomic Skills
Instead of training a single 300-step policy, we train separate, short-horizon **atomic skills** (taking 30–50 steps):
*   **Skill A**: *"Reach and open drawer"*
*   **Skill B**: *"Pinch and lift cube"*
*   **Skill C**: *"Place cube in drawer"*

Because the horizon for each skill is short, the reward signal is dense, and training converges in under 5 hours of L4 GPU time.

### 2. State-Tracking via the Memory Triad
Instead of a complex VLM sequencing actions, the **Memory Triad** coordinates the execution loop programmatically:
*   **Event Boundary Anchors**: Detect physical landmarks (e.g., the tactile contact onset of opening the drawer). Once the landmark is reached, the memory system registers the boundary and triggers the next skill in the library.
*   **Long-Range Gist Tokens (HyDRA)**: Maintain spatial object permanence. If the drawer or block is temporarily occluded by the robot's arm, the memory system recalls the coordinates from the Gist tokens, ensuring the active policy does not suffer from visual memory fade.
*   **PSN Skill Library**: Organizes the trained weights for each atomic skill. During downstream execution, the robot retrieves the relevant weights based on the active state boundary.

This decoupling ensures the neural networks only handle high-frequency local control, while the memory architecture manages the long-horizon sequencing, compensating for our limited compute budget.

---

## Detailed Training Stage Blueprint

The following blueprint details the sequence of training stages, mapping the architectural components, input/output data, objective functions, and specific outcomes for each phase.

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Latent Dynamics Pre-training                                  │
│ (Action-Agnostic / Video-Scale / Frozen Checkpoints)                   │
│  - Encoders (DINOv3, CLIP, PointNeXt, VGGT) & JEPA World Predictor     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Supervised Fine-Tuning (SFT) & Action Mapping                 │
│ (Embodiment-Specific / Joint-Level / Learnable Adapters)               │
│  - Flow-Matching Action Denoiser, MLP Adapters, pycapacity Polytope    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: RL Alignment & Unsupervised Exploration                       │
│ (On-Policy / Simulation-Rich / Skill Chaining)                         │
│  - RAM, Discriminator-Guided RL, d-OPSD, EBT Safeguards, RATs          │
└────────────────────────────────────────────────────────────────────────┘
```

| Stage & Active Components | Inputs / Outputs | Objectives & Losses | Outcomes | Validation Checks (Gatekeepers) |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Pre-training** *(Zero-Shot prior)*<br><br>**Active Components**:<br>• Frozen Encoders (DINOv3, CLIP, PointNeXt, VGGT)<br>• JEPA Latent Predictor | **Inputs**: Raw video streams (human + robot), multi-view static frames, point clouds.<br>**Outputs**: Target and predicted latent states ($z_t, \hat{z}_{t+1:t+H}$). | • **Zero-Shot Encoders**: Weights loaded from public pre-trained checkpoints (DINOv3, CLIP, VGGT).<br>• **JEPA Latent Predictor Loss**: Minimizes next-step latent transition error $D(\hat{s}_y, s_y)$ over the frozen representation space. | • Builds a structured latent dynamics manifold ("directional highway").<br>• Establishes zero-shot priors for spatial waypoints, visual object interaction, and language instruction grounding. | • **"No-Op" Loss Ratio** $< 0.2$ (verifies the dynamics model isn't copy-pasting).<br>• **Action Perturbation Drift** $> \delta$ (verifies predictor is sensitive to action inputs).<br>• **Multi-Step Rollout MSE** remains stable up to $H=10$. |
| **Stage 2: SFT / Mid-training** *(Action Grounding)*<br><br>**Active Components**:<br>• Flow-Matching Action Denoiser (CLAP-RF)<br>• Trainable MLP Adapters<br>• Tactile/Proprioceptive stream<br>• `pycapacity` Task-Space filter | **Inputs**: Latent states ($z_t$), target spatial waypoints, joint angles/torques, tactile arrays.<br>**Outputs**: Generated multi-step joint trajectory ($a_{t:t+H}$). | • **Conditional Flow Matching (CFM) Loss**: Regresses predicted velocity fields $v_\theta$ to match target trajectories.<br>• **Adapter Mapping Loss**: Trains lightweight MLPs to map DINO/CLIP features into robot dynamics space. | • Binds the abstract world model dynamics to the concrete physical embodiment.<br>• Instructs the model how to propose kinematically feasible trajectories within workspace polytope boundaries. | • **CFM Val Loss** converges below target threshold.<br>• **Workspace Polytope Violation Rate** $= 0\%$ (using the `pycapacity` filter).<br>• **Basic Reaching Success** $\ge 80\%$ (open-loop in sim). |
| **Stage 3: RL & Exploration** *(Behavior Tuning)*<br><br>**Active Components**:<br>• RAM (Reinforce Adjoint Matching)<br>• Discriminator ($D_\psi$)<br>• d-OPSD / PF-OPSD<br>• EBT-Policy (MCMC / Langevin)<br>• RATs Task Proposer | **Inputs**: On-policy simulator states, dense progress rewards, adversarial perturbations.<br>**Outputs**: Fine-tuned flow fields, programmatic skills in library. | • **RAM Velocity Correction Loss**: Direct regression weighting of flow based on scalar rewards.<br>• **Contrastive Energy Loss**: Minimizes prediction error for successes; maximizes ($E \to \infty$) for failures. | • Aligns policies to perform high-precision grasp, pinch, and lift operations.<br>• Develops robust recovery skills from catastrophic failures through self-distillation.<br>• Populates the latent skill library autonomously. | • **Custom Task Success Rate** $\ge 90\%$ (grasp, lift, place success).<br>• **Energy Delta** (Clear separation: $E_{\text{fail}} \gg E_{\text{success}}$).<br>• **d-OPSD Recovery Rate** $\ge 90\%$ under active physical perturbations. |

---

## 1. Stage 1: Latent Dynamics Pre-training (Action-Agnostic)

To leverage large-scale external repositories (such as those hosted on `datasets.bot` or Hugging Face LeRobot) which lack joint values, we decouple the training into **dynamics pre-training** and **embodiment-specific action mapping**.

For human videos or diverse robot trajectories lacking joint positions, we employ the **VLA-JEPA** paradigm:
* **The Objective**: Predict the future latent state $\hat{z}_{t+1:t+H}$ from current visual observations $z_t$.
* **The Supervision**: The target is provided by the Encoder Unit processing the future frames. Because the student backbone is not allowed to see the future frames, it is forced to construct a representation of the *latent action* via the Latent Action Encoder ($h_\psi$) representing the visual vector field of the environment's evolution.
* **Pre-trained Encoders**: We freeze semantic encoders like **DINOv3** (for dense visual representations / full-frame features) and **CLIP** (for high-level task/semantic conditioning).

---

## 2. Stage 2: Supervised Fine-Tuning (SFT) & Action Grounding

Once the world model understands latent transitions, we introduce the embodiment-specific joint command dataset.

* **Expert Dataset Grounding**: We introduce our embodiment-specific teleoperated dataset (sourced from `datasets.bot` or Hugging Face LeRobot, e.g., the AgiBot or Astribot tabletop trajectories).
* **Action Adapter Mapping**: The **CLAP-RF Flow Matching model** is trained to map these latent transitions ($z_t \rightarrow z_{t+1}$) to the specific joint values (actions/torques) of the target humanoid embodiment via the Action Adapter ($f_{\text{action}}$).
* **Unified Limits**: For transferability across different robot setups in `datasets.bot`, we map joint spaces into a unified **Task-Space Polytope representation** (via `pycapacity` Cartesian 6DoF limits) during pre-training, only mapping to low-level actuator torques during final post-training.
* **ComboStoc Asynchronous Training**: During CFM training, instead of adding synchronous noise to all attributes of a sample, we utilize the **ComboStoc (arXiv:2405.13729)** paradigm by vectorizing flow steps asynchronously across different input dimensions (e.g. vision, text, joints, tactile). This trains the model to perform highly robust conditional flow generation, enabling it to clean up noisy modal inputs (like occluded vision or unstable touch feedback) by conditioning on the cleaner, aligned dimensions.

---

## 3. Stage 3: RL Alignment, Energy Sculpting & Unsupervised Exploration

Once Stage 1 (Dynamics) and Stage 2 (SFT Joint Grounding) are complete, the robot possesses a baseline model of physical transition dynamics and rudimentary coordination. The **Stage 3 RL & Unsupervised Exploration** phase aims to optimize this coordination into a robust, collision-free, and adaptive policy.

The RL training loop operates via four primary mechanisms:

1. **Active Exploration (The Play Loop)**:
   * The robot runs in simulation, attempting tasks proposed by the Task Proposer (or target keypoints).
   * It receives dense progress rewards based on the distance between the current state $s_t$ and target boundary latents, along with sparse rewards when it crosses the event boundary.
2. **Adversarial Trajectory Shaping**:
   * A discriminator ($D_\psi$) is trained concurrently to distinguish simulated trajectories from human-teleoperated trajectories.
   * By feeding the discriminator's gradient as a reward to the policy, we ensure that the explored trajectories remain smooth, natural, and physically consistent, avoiding erratic RL movements.
3. **Contrastive Energy Landscape Sculpting**:
   * The robot will inevitably encounter failures (e.g. joint locks, object slips, collisions) during exploration.
   * These failures are logged into the Memory Triad. The JEPA predictor is updated contrastively: minimizing prediction energy for successful paths and maximizing it ($E \to \infty$) for failures.
   * This carves deep energy "valleys" around successful coordinates and builds high-energy "walls" around failure zones. The policy's planner can use the gradient $\nabla_a E$ to steer proposed actions away from failure regions on-the-fly.
4. **d-OPSD Suffix-Conditioned Self-Distillation**:
   * When the robot successfully recovers from a near-miss or slip, the full recovery trajectory is passed to a teacher model.
   * The teacher generates the optimal flow field conditioned on the full recovery sequence (suffix conditioning).
   * The student policy (CLAP-RF) is trained to match the teacher's flow using *only* the start state $s_t$, embedding robust, zero-latency recovery reflexes directly into the continuous control loop.

---

### A. Generator-Discriminator Duo (Flow Matching & EBM)

The integration of Flow Matching and JEPA resolves the limitations of blind CEM/MPC sampling:

```
  ┌───────────────────────────────────────────────────────────────┐
  │ 1. THE GENERATOR (CLAP-RF Flow Head)                          │
  │    - Generates joint action trajectories from noise.          │
  └────────┬──────────────────────────────────────────────▲───────┘
           │ (Proposed Trajectory)                        │ (Adversarial Reward)
           │                                              │
  ┌────────▼─────────────────────────┐           ┌────────┴─────────────────────┐
  │ 2. THE EBM (JEPA Predictor)      │           │ 3. DISCRIMINATOR ($D_\psi$)  │
  │    - Evaluates physical energy   │           │    - Evaluates human-like    │
  │      (L2 prediction error).      │           │      trajectory realism.     │
  └──────────────────────────────────┘           └────────▲─────────────────────┘
                                                          │ (Reference Demos)
                                                 [ datasets.bot Expert Demos ]
```

* **Interactive Latent Generation (DAWN Reciprocity)**:
  Instead of a sequential predict-then-plan approach, the models run a recurrent loop:
  1. **Flow Proposal**: The Action Denoiser generates a noisy action trajectory $a^{(k)}_{t:t+H}$ using continuous flow matching.
  2. **Predictive Rollout**: The JEPA World Predictor rolls out the latent future $\hat{z}_{t+1:t+H}$ conditioned on $a^{(k)}$.
  3. **EBM Evaluation**: The JEPA predictor evaluates the energy $E(z_t, a, \hat{z})$—representing physical viability and semantic progress. 
  4. **Adjoint Guidance**: The gradient of this energy $\nabla_a E$ is fed back to guide the next flow step $a^{(k-1)}$ toward high-reward, low-energy configurations.

* **Adversarial Perturbations & Sample Generation**:
  The Generator (CLAP-RF), EBM (JEPA), and Discriminator ($D_\psi$) are bound together by **Task Conditioning ($z_{\text{task}}$)** derived from user textual instructions (e.g. *"pinch the block"*) via the CLIP Text encoder and MLP adapters.
  * **Real vs. Fake Discriminator Pairs**: The Discriminator ($D_\psi$) is trained to distinguish whether a trajectory $\tau$ is a *real* human demonstration of task $z_{\text{task}}$ (sourced from datasets.bot, labeled $1$) or a *fake* simulated trajectory generated by CLAP-RF for the same task $z_{\text{task}}$ (labeled $0$). The Generator is updated to maximize this score, driving it to mimic natural, human-like motion.
  * **BadWorld Adversarial Failure Generation**: To prevent EBM "reward hacking" (where the policy exploits flat regions of the predictor's dynamics model), we actively generate adversarial failure samples in the simulator. During rollouts for task $z_{\text{task}}$, we inject structured force/velocity perturbations to the joint torques. This forces physical failures (slips, misses, collisions). The EBM (JEPA) evaluates these transitions and is trained to assign them **high energy ($E \to \infty$)**, mapping the physical limits of the task.
  * **Flow Reversal Steering (FRS)**: When the Generator outputs a trajectory that leads to a failure, we run the flow matching ODE in reverse ($t=1 \to t=0$) to find the exact starting noise vector ($x_0$) that caused the failure. We then update the vector field to steer $x_0$ away from the failure mode and toward a successful trajectory, correcting the policy directly in the noise space.

* **π-StepNFT (Step-wise Negative-Aware Fine-Tuning)**:
  To fine-tune the CLAP-RF flow model online without a complex critic network, we implement the **π-StepNFT (arXiv:2603.02083)** framework:
  1. **Step-wise Denoising Supervision**: Instead of evaluating the policy only at the end of the action trajectory, we apply fine step supervision at every intermediate denoising step of the ODE integration.
  2. **Negative-Aware Gradients**: When the Generator drifts off-manifold into failure modes (identified by the EBM or discriminator), we calculate negative-aware step updates that actively push the flow velocity fields $v_\theta(a_t, t)$ away from the failed paths at that specific step $t$. This forces the flow trajectories to explore wider, safe boundaries around the expert path.

### B. Unsupervised Skill Discovery & On-Policy Self-Distillation (OPSD)

While the Adversarial RL loop (Stage 3 A) optimizes the policy's style and target-tracking in real-time, the **Unsupervised Skill Discovery & d-OPSD** phase acts as an offline **distillation and recovery compiler**. It discovers new behaviors during unsupervised play and distills complex multi-step recovery maneuvers into reactive, zero-latency policy weights.

```
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. PLAY PHASE (RATs Exploration in Simulator)               │
  │    - Robot attempts random targets to discover skills       │
  └────────┬────────────────────────────────────────────────────┘
           │
           ▼ (Completed Trajectories: Successes & Near-Miss Recoveries)
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. TEACHER MODEL (Suffix-Conditioned Denoising)             │
  │    - Ingests the ENTIRE completed trajectory (suffix).      │
  │    - Computes the optimal flow velocity field (v_teacher).  │
  └────────┬────────────────────────────────────────────────────┘
           │
           ▼ (Velocity Targets)
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. STUDENT POLICY (CLAP-RF Controller - Stage 3 Policy)     │
  │    - Ingests ONLY the current starting state (st).          │
  │    - Trained to regress and match v_teacher.                │
  └─────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    [ Populates PSN Skill Library ]
```

* **d-OPSD (Diffusion Suffix Conditioning)**:
  * **The Problem**: If the robot experiences a physical disturbance (e.g. hand slips off the block), standard RL requires the policy to dynamically search for a recovery path online. This is slow and often fails in high-frequency control.
  * **The Solution**: d-OPSD reformulates self-distillation for flow policies. When the robot successfully recovers from a slip during play, the *entire completed trajectory* (the success suffix) is fed to a high-capacity **Teacher Model**. Because the teacher sees the future outcome, it easily calculates the optimal, smooth velocity field $v_{\text{teacher}}$ that resolved the slip.
  * **Student Distillation**: The **Student Model (CLAP-RF)** only has access to the current state $s_t$ (no future suffix). We train the student to match the teacher's predicted flow velocity $v_{\text{teacher}}$ at every step: $\mathcal{L}_{\text{OPSD}} = \| v_{\text{student}}(s_t, t) - v_{\text{teacher}}(s_t, t \mid \text{suffix}) \|^2$. This compiles complex recovery maneuvers directly into the student's feedforward weights.

* **Playful Exploration (RATs)**:
  * Operating in MuJoCo, the **Task Proposer** selects semantic targets in the DINOv3 space.
  * **Play Phase**: The humanoid attempts trajectories generated by the Flow Matcher.
  * When a trial succeeds, the trajectory is distilled into the **Programmatic Skill Network (PSN)** as an executable symbolic macro, then stored in the memory library.

### C. Policy Bootstrapping & Stagnation Prevention (Auto-NPO)

To prevent the on-policy RL loop from hitting optimization plateaus or stagnating in simulation when exploring complex tabletop setups, we integrate the **Auto-NPO (arXiv:2604.20733)** framework directly into the Stage 3 training scheduler.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. CURRENT POLICY (CLAP-RF Checkpoint t)                        │
  │    - Low success rate / Stagnated on complex contact task       │
  └────────┬────────────────────────────────────────────────────────┘
           │                                          ▲
           ▼ (Evaluates style gap V)                  │ (Bootstrapped S=Q/V updates)
  ┌───────────────────────────────────────────────────┼─────────────┐
  │ 2. AUTO-NPO SELECTION SCHEDULER                   │             │
  │    - Queries future checkpoints (t + Δt)          │             │
  │    - Maximizes signal: S = Q (Success) / V (Style Gap)          │
  └────────────────────────┬──────────────────────────┘             │
                           │                                        │
                           ▼ (Selects optimal future self)          │
  ┌─────────────────────────────────────────────────────────────┐   │
  │ 3. FUTURE POLICY SELF (CLAP-RF Checkpoint t + Δt)           │   │
  │    - Generates high-quality demonstrations (Q)              │───┘
  │    - Zero style mismatch (V ≈ 0)                            │
  └─────────────────────────────────────────────────────────────┘
```

* **The Stagnation Problem**: Standard RL updates the policy based strictly on on-policy exploration. In high-dimensional humanoid control, if the active policy cannot find a successful trajectory, the policy gradient collapses. Alternatively, trying to force the policy to mimic human trajectories directly during online RL introduces a significant style mismatch (distribution gap), slowing down convergence.
* **The Auto-NPO Mechanism**: Instead of relying on distant external expert trajectories, we utilize a **slightly more advanced future checkpoint** of our policy from a parallel training branch. 
  1. We mix rollout trajectories from the model's "near-future self" into the current policy's update group.
  2. The update scheduler dynamically selects the future checkpoint that maximizes the learning signal ratio: $S = Q/V$, where $Q$ is the trajectory success rate (quality) and $V$ is the KL divergence between the current policy and the future checkpoint (style variance).
  3. Because the future checkpoint shares the exact same morphology and movement style as the current policy, the style variance ($V$) is minimized, allowing the current policy to easily absorb the training trajectories and breakthrough learning plateaus.

---

## 4. The Hybrid Memory Triad (MemoryWAM & Echo-Memory)

To maintain object permanence and stable skill recall during continuous humanoid manipulation (e.g., keeping track of a block's position even when it is fully occluded by the robot's hand during a pinch), we implement a multi-tiered **Hybrid Memory Triad**:

```
 [ Raw Modal Inputs ] ──► [ ENCODER UNIT ]
                                │
                                ├─────────────────────────┐
                                │ (10Hz Live Tokens)      │ (Salience Triggers)
                                ▼                         ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ HYBRID MEMORY TRIAD                                           │
  │                                                               │
  │  1. SHORT-TERM WINDOW (Sliding Window: Last 5-10 Steps)       │
  │     - Raw uncompressed visual/tactile tokens.                 │
  │                                                               │
  │  2. EVENT BOUNDARY ANCHORS (Full spatial latents at triggers) │
  │     - Locked at contact-rich milestones (e.g., initial touch).│
  │                                                               │
  │  3. LONG-RANGE GIST (HyDRA Tokens: Compressed History)        │
  │     - Summarized history preserving spatial coordinates.      │
  └─────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼ (Fused Memory Context Tokens)
                    [ CLAP-RF / Predictor ]
```

### The Three Tiers of Memory Routing

1. **Short-Term Context (Sliding Window)**:
   * **Mechanism**: A fast-access buffer that stores uncompressed visual and tactile tokens from the last $5$ to $10$ steps ($0.5$–$1.0$ seconds of history).
   * **Purpose**: Provides immediate, local temporal context for high-frequency (10Hz+) reactive physics adjustments, such as micro-slippage corrections during the pinch phase.

2. **Event Boundary Anchors (Full Spatial Latents)**:
   * **Mechanism**: Triggered by high-salience physical milestones (e.g., tactile pressure spike at contact, force sensor drop at object release). At the boundary step, we freeze and store the complete, uncompressed spatial latent state.
   * **Purpose**: Serves as a persistent anchor for the task target, ensuring the model's goal representation does not drift over long horizons.

3. **Long-Range Gist (Relevance-Driven HyDRA Tokens)**:
   * **Mechanism**: Historical tokens are compressed using a relevance-based attention bottleneck (HyDRA tokens) to summarize the past.
   * **Purpose**: Maintains object permanence. If an object is temporarily occluded by the robot's own arm or moved out of the egocentric camera field, the Gist tokens retain its spatial coordinates, preventing "visual memory fade."

### Skill Chaining & Composability via Event Boundaries
Rather than using memory solely to avoid visual drift, the Memory Triad acts as a **composability coordinator** for skills:
* **Sequential Skills**: Complex long-horizon behaviors (e.g., drawer-opening followed by picking an object inside) are composed of primitive skills. The transition between skills is managed programmatically via **Event Boundary Anchors** (physical landmarks like a tactile contact spike or position milestone).
* **Composition Routing**: When a milestone triggers, the memory system indexes the next skill token from the **Programmatic Skill Network (PSN)** library, swapping the conditioning input of the flow matcher on-the-fly. This allows modular, zero-shot skill chaining without needing to train a monolithic, end-to-end policy for every new task combination.

### Offline Consolidation (Sleep Phase)
* **Knowledge Seeding**: Distills the fragile, short-term exploratory trajectories captured in the memory triad during the day into the core weights of the dynamics predictor and adapters.
* **Dreaming (Latent Rehearsal)**: Uses **Reinforce Adjoint Matching (RAM)** over stored Gist and Event Boundary anchors to generate synthetic rollout curriculums. This allows the robot to rehearse tasks offline in its own imagined latent space, saving simulator compute.

---

## 5. Broad Research Vision: User-Guided Rapid Skill Adaptation

The ultimate goal of this architecture is to move away from rigid, pre-defined robotic task execution and enable **zero-friction, on-the-fly skill discovery and deployment**. 

Under this vision, a user can walk up to the robot, define a task conceptually, and watch the system rapidly learn, stabilize, and export the new skill vector without intensive manual data collection.

```
                  USER INPUT (Language / Keypoint Images)
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 1. TASK GROUNDING & EVENT BOUNDARY CONFIGURATION    │
        │    - User sets visual anchors (goal state keypoints)│
        │    - System creates dynamic reward detector         │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 2. SAMPLE-EFFICIENT LOCAL EXPLORATION (RL STAGE 3)  │
        │    - Anchored by Stage 1 Dynamics & Stage 2 Motors  │
        │    - System runs rapid, local trial-and-error       │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 3. MEMORY CONSOLIDATION & SKILL EXPORT              │
        │    - Successes saved as Event Boundary / Gist tokens│
        │    - Distilled into PSN vector macro and exported   │
        └─────────────────────────────────────────────────────┘
```

### A. How and When Skills are Developed in the Training Pipeline
* **Stage 1 (Dynamics Pre-training)**: The system learns **zero task-specific skills**. It focuses entirely on physical dynamics and object behaviors (e.g., how drawers open, how cubes behave when pushed) by observing diverse robot/human videos.
* **Stage 2 (SFT Action Grounding)**: The system learns **primitive coordination macros** (e.g., reaching, opening, picking) from teleoperated data. This populates the Programmatic Skill Network (PSN) with standard baseline weights.
* **Stage 3 (RL / Rapid Customization)**: The system learns the **specific target skill**. Using the Stage 1 and Stage 2 priors, the model runs highly focused, sample-efficient local RL in simulation to bridge the gap from standard coordination to the user's customized task.

### B. User-Guided Keypoint Anchoring (Future-Proofing the Design)
To support this final demo, the Event Boundary Anchors in our Memory Triad are designed to be **digitally configurable**:
1. **Defining the Boundary (SAM Point Cloud Filtering)**: The user interacts with the UI camera feed and clicks on the target object (e.g., the block or the mug). The **Segment Anything Model (SAM)** runs offline on this click, generating a 2D mask. We use this mask to filter the raw point cloud, cropping out everything except the target coordinates. This cropped point cloud is passed through PointNeXt, allowing the user to configure precise visual/spatial keypoints (e.g., matching the block's coordinate tokens to the mug's coordinate tokens in PointNeXt projection space) without background clutter.
2. **Dynamic Reward Construction**: These keypoints are used to dynamically configure the Reward Discriminator ($D_\psi$) and the Event Boundary Anchor. The robot knows the task is complete the moment the sensor/visual representations match the keypoint boundaries.
3. **Exploratory Trials**: The robot attempts multiple approaches. When a trial successfully crosses the user's defined Event Boundary, the success trajectory is recorded in the Memory Triad, distilled via d-OPSD, saved as an executable symbolic vector in the PSN, and exported as a standalone macro.

---

## 6. Next Steps & Implementation Roadmap

To execute this architecture systematically, we divide the roadmap into four developmental phases:

### Phase 1: Proof-of-Concept (PoC) Validation

To verify our core mathematical formulations before setting up the full training environment, we will execute five self-contained validations:

*   **Task 1.1: CLAP Flow Matcher Validation (`flow_matcher_poc.py`)**:
    *   *Objective*: Verify the vector field regression and trajectory generation capability of the CLAP-RF head.
    *   *Implementation*: Regress velocity fields on expert actions and test Euler integration sampling under asynchronous noise schedules (ComboStoc).
    *   *Gatekeeper Success Metric*: Flow trajectories must converge to target endpoints in $< 20$ integration steps without path divergence.
*   **Task 1.2: Pretrained Encoders Visualization (`visualize_encoders_poc.py`)**:
    *   *Objective*: Verify the outputs and feature characteristics of CLIP, DINOv3, VGGT, SAM, and PointNeXt.
    *   *Implementation*: Render cosine similarity maps (CLIP), self-attention keypoints (DINO), camera ego-motion tracks (VGGT), visual segmentation masks (SAM), and cropped coordinate arrays (PointNeXt).
    *   *Gatekeeper Success Metric*: All visualizations must yield correct semantic highlighting, correct depth tracks, and clean coordinate bounds without background bleed.
*   **Task 1.3: Latent Action Encoder & MSAT Validation (`msat_latent_action_poc.py`)**:
    *   *Objective*: Verify state-transition compression and cross-modal attention routing.
    *   *Implementation*: Compress transitions ($s_t \to s_{t+1}$) using the shallow Latent Action Encoder helper ($h_\psi$) and test MSAT cross-attention weighting.
    *   *Gatekeeper Success Metric*: Visualized MSAT attention maps must dynamically route focus to contact modalities (tactile) upon impact and visual modalities during approach, with $0\%$ state identity leakage from $h_\psi$.
*   **Task 1.4: Predictor Grounding & Sensitivity (`predictor_poc.py`)**:
    *   *Objective*: Prevent predictor copy-paste collapse ($\hat{s}_{t+1} \approx s_t$) and test rollout stability.
    *   *Implementation*: Evaluate prediction error across multi-step rollouts ($H > 1$) and compute action-sensitivity gradients ($\nabla_a E$).
    *   *Gatekeeper Success Metric*: Maintain a high action perturbation drift ($\Delta_{\text{action}} > 0.1$) under modified action inputs, ensuring the world model remains highly sensitive to actions.
*   **Task 1.5: Tactile Adapter Calibration (`tactile_adapter_poc.py`)**:
    *   *Objective*: Project and align simulated touch sensor outputs into the unified token space.
    *   *Implementation*: Extract $4 \times 4$ normal force arrays from MuJoCo `sensordata` and align them with vision using the InfoNCE-based contrastive loss (CASA).
    *   *Gatekeeper Success Metric*: Adapter successfully filters low-force sensor noise ($< \epsilon$) and establishes high cosine similarity alignment during contact events.

### Phase 2: Data Selection & Preparation
* **Task 2.1: datasets.bot Audit**: Select a diverse, multi-task tabletop manipulation dataset (e.g., AgiBot or Astribot tabletop trajectories) containing human-teleoperated episodes.
* **Task 2.2: Modal Tokenization & Caching Pipeline**: Write the preprocessing script (`preprocess_dataset.py`) to pass images, point clouds, and text instructions through frozen DINOv3, CLIP, PointNeXt, and VGGT backbones. If raw point clouds are missing (e.g. on 2D video datasets), the pipeline estimates monocular depth maps and back-projects them into 3D camera space to extract PointNeXt features.

### Phase 3: SFT Training Loop Implementation
* **Task 3.1: MLP Adapter & MSAT Coding**: Write the PyTorch code for the trainable MLP adapters and the Multi-Stream Action Transformer (MSAT) cross-attention layers.
* **Task 3.2: CFM Action Denoiser Loop**: Implement the Conditional Flow Matching (CFM) training script for the CLAP-RF denoiser, projecting outputs through the `pycapacity` Task-Space filter.

### Phase 4: RL & Dreaming Pipeline
* **Task 4.1: Reinforce Adjoint Matching (RAM)**: Code the latent imagination loop, allowing CLAP-RF to rehearse trajectories inside the frozen JEPA predictor space.
* **Task 4.2: Contrastive Sculpting & d-OPSD**: Implement the contrastive energy updater and the suffix-conditioned recovery distillation loop.

---

## 7. Verified Proof-of-Concept (PoC) Implementations

To ensure mathematical and computational correctness before full training, we have implemented and validated six architectural PoC scripts inside the `poc/` directory:

1. **`flow_matcher.py` (Flow Matcher)**:
   * *Status*: `SUCCESS` (Euler integration trajectory convergence error: `0.0628`).
   * *Validation*: Confirmed that the flow-matching head learns smooth continuous vector fields and reconstructs expert action paths within $20$ steps.
2. **`visualize_encoders.py` (Visual Encoders)**:
   * *Status*: `SUCCESS` (Verified DINOv3, SAM 2, VGGT, CLIP, and depth maps).
   * *Validation*: Implemented CLS-to-Patch Cosine Similarity for DINOv3 attention map, Lucas-Kanade optical flow camera tracking, and SAM 2 target point mask projections.
3. **`msat_latent_action.py` (MSAT Cross-Attention & Latent Action)**:
   * *Status*: `SUCCESS` (Verified 0% state identity leakage).
   * *Validation*: Confirmed that the Latent Action Encoder ($h_\psi$) compresses state-transitions into abstract actions without passing raw state identity. Verified that MSAT cross-attention weights dynamic modality priorities.
4. **`predictor.py` (JEPA Dynamics)**:
   * *Status*: `SUCCESS` (Verified collapse prevention).
   * *Validation*: Evaluated multi-step rollouts and proved that bilinear action-state gating forces action sensitivity, preventing predictor copy-paste collapse.
5. **`tactile_adapter.py` (Tactile Adapter)**:
   * *Status*: `SUCCESS` (Verified InfoNCE alignment).
   * *Validation*: Projects spatial contact pressure inputs into the unified $512$-dim token space, aligning visual contact with tactile contact events.
6. **`pycapacity_test.py` (Safety Filter Polytope)**:
   * *Status*: `SUCCESS` (Projection latency $< 1\text{ms}$).
   * *Validation*: Solved the 69,000-constraint vertex bottleneck by formulating a direct 24-constraint H-polytope representation of joint limits, ensuring real-time projection speed.

---

## 8. Integrated Pre-Training & SFT Datasets

To train and fine-tune LatentFlow on standard Google Colab instances without disk memory exhaustion, we employ a highly specific, lightweight **5-10 GB tabletop pre-training dataset mix** coupled with selective single-episode downloads:

### A. Stage 1: Transition Dynamics Pre-training (60:30:10 Mix)
*   **60% Tabletop Robot Manipulation (Droid)**:
    *   *Dataset*: **`lerobot/droid`** (standardized in LeRobot format, aligned with the V-JEPA 2-AC action-conditioned world model).
    *   *Role*: grounds the dynamics predictor in robot joint space constraints and physical table-top transitions.
*   **30% Human Tabletop Hand-Object Interaction (EgoScale & E2E-3M)**:
    *   *Dataset*: **`lerobot/cmu_stretch`** and egocentric VQA clips from Ego4D.
    *   *Role*: Teaches hand-object affordances, tool-use semantics, and visual contact priors from first-person human actions.
*   **10% 3D Visual Geometry & Tracking (PointOdyssey)**:
    *   *Dataset*: PointOdyssey coordinate point tracking and synthetic table blocks.
    *   *Role*: Anchors predictor transitions in strict 3D topological tracking and geometric camera transformations.

#### Modality Mapping and Processing Rules:
The table below highlights what modalities are present in each dataset stream, and how they are handled during Stage 1 pre-training:

| Modality / Step | Stream A: Droid (Robot) | Stream B: CMU Stretch (Human) | Stream C: PointOdyssey (Geometry) |
| :--- | :--- | :--- | :--- |
| **2D Video Frames (RGB)** | **Present** (Franka cameras) | **Present** (Egocentric camera) | **Present** (CG camera render) |
| **Text Instruction** | **Present** (H-L Task prompt) | **Present** (H-L Task prompt) | **Present** (H-L Task prompt) |
| **Ground-Truth 3D Coordinates** | **Missing** | **Missing** | **Present** (Saved on disk as `point_clouds.npy`) |
| **Actions (Robot joints)** | **Present** (Franka joint torques) | **Missing** (Zero-padded / Zero-masked) | **Missing** (Zero-padded / Zero-masked) |
| **Proprioception (Robot state)** | **Present** (Franka joint angles) | **Missing** (Zero-padded / Zero-masked) | **Missing** (Zero-padded / Zero-masked) |
| **Tactile Grid (Touch)** | **Missing** (Zero-padded / Zero-masked) | **Missing** (Zero-padded / Zero-masked) | **Missing** (Zero-padded / Zero-masked) |
| **3D Generation Method** | **Calculated** (SAM + Depth + KLT tracking) | **Calculated** (SAM + Depth + KLT tracking) | **Direct Load** (Loaded directly from `point_clouds.npy`) |

### B. Stage 2: Supervised Fine-Tuning (SFT) & Action Grounding
*   **Bimanual ALOHA Tabletop (`lerobot/aloha_mobile_cabinet`)**:
    *   Provides paired camera frames and joint torque/velocity commands.
    *   *Role*: Trains the Flow-Matching action head and Action Adapter to command coordinated joint trajectories.
*   **UT Austin Sort & NYU Door/Drawer Opening (OXE)**:
    *   Provides specific demonstration trajectories for target pick-and-place and sliding-door operations.

### C. Stage 3: RL & Tactile Alignment
*   **Genesis/MuJoCo Simulation Rollouts**:
    *   *Role*: On-policy active exploration collecting positive successes and negative failures for contrastive energy landscape sculpting.
*   **Tactile-Dexterous Real Touch Datasets**:
    *   *Role*: Aligns tactile pressure maps with visual contact boundaries via InfoNCE.


## Execution Quickstart Reference

Here are the standard commands to run the data preparation, sampling, and training modules.

### 1. Dataset Preparation & Mixing
Downloads and preprocesses the multi-modal pre-training mixture.
* **CPU / Mock Mode Run (Dry-run)**:
  ```bash
  PYTHONPATH=latent-flow python latent-flow/utils/prepare_and_visualize_dataset.py
  ```
* **GPU Run (Extracts real foundation features on CUDA)**:
  ```bash
  PYTHONPATH=latent-flow python latent-flow/utils/prepare_and_visualize_dataset.py --enable_encoders --clean
  ```

### 2. Multi-Modal Verification & Zip Packaging
Compiles raw image sequences into VS Code compatible VP9 MP4 videos, audits processed representation tensors, and packages files for local inspection.
```bash
python latent-flow/utils/sample_and_package.py --raw_dir latent-flow/data/raw --processed_dir latent-flow/data/processed --output_zip dataset_samples.zip
```

### 3. Stage 1 Latent Dynamics Pre-training
Optimizes transition predictive dynamics inside the V-JEPA transformer core, logging checkpoints and curves to PyTorch Lightning and W&B.
* **Diagnostic Check (Subset of 5 episodes, fast verification)**:
  ```bash
  python latent-flow/train.py --stage 1 --use_subset --config latent-flow/config/default_config.yaml
  ```
* **Full Pre-training Run**:
  ```bash
  python latent-flow/train.py --stage 1 --config latent-flow/config/default_config.yaml
  ```


