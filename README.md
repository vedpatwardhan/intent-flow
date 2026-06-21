## LeWM-Flow

## 1. Empirical Baseline and the Limits of Latent Model-Predictive Control

The pursuit of robust, closed-loop dexterous manipulation in high-degree-of-freedom (DoF) humanoid platforms remains a formidable challenge in embodied artificial intelligence. Recent empirical audits of the LeWorldModel (LeWM) architecture, applied to a Fourier GR-1 humanoid executing a tabletop cube pickup in a MuJoCo simulation environment, have exposed critical vulnerabilities in traditional latent Model-Predictive Control (MPC) mechanisms. The baseline manipulation task was segmented into 32-frame episodes operating at 10Hz, rigorously divided into four distinct 8-frame sub-phases: approach, pre-grasp alignment, pinch, and lift.

The integration of representation learning advancements-specifically multi-view spatial fusion, skeletal geometric priors, and DINOv3 waypoint supervision-successfully restructured the latent training manifold of the Joint-Embedding Predictive Architecture (JEPA). Through dimensionality reduction techniques (PCA, t-SNE, UMAP), the latent topology transitioned from isolated, highly entangled "worms" characteristic of single-view RGB baselines, into a structured, directional "highway". This geometric organization indicated that the JEPA encoder and predictor successfully acquired a temporal and spatial understanding of the 200 teleoperated expert trajectories.

However, despite this highly organized global embedding space, downstream closed-loop planning via the Cross-Entropy Method (CEM) proved brittle and functionally inadequate. Latent MPC frameworks operate by sampling massive quantities of raw action sequences, rolling the predictor forward in the latent space, and scoring the generated trajectories against a progress reward (the inverse distance from the right-hand fingertips to the cube). In practice, this random-shooting optimization searches blindly within the raw action space. While the humanoid successfully approached and established initial contact with the cube, the contact-rich pinch and lift phases consistently failed due to off-manifold latent drift and the generation of out-of-distribution (OOD) kinematics.

To mitigate the physical absurdity of CEM proposals, heuristic kinematic bounding via task-space polytopes using the pycapacity library was introduced. The demonstrated workspace polytope acts as a rigid containment field, calculating real-time task-space velocity ($\mathcal{P}_v$) and force ($\mathcal{P}_f$) limits derived from the humanoid's joint encoders and the Jacobian matrix. While these feasibility filters rejected catastrophic kinematic anomalies (such as inverted elbow bending or self-collisions), they merely masked the fundamental algorithmic deficit: CEM lacks a generative action prior capable of intrinsically proposing temporally coherent and physically grounded trajectory distributions.

Furthermore, relying exclusively on an initial dataset of 200 pure-success expert demonstrations inherently crippled the model's understanding of the broader state-action manifold. The subsequent acquisition of an expanded 2000-episode dataset-comprising 500 successful pickups, 1000 sub-optimal near-misses, and 500 randomized catastrophic failures-necessitates a fundamental architectural shift. The objective is to discard heuristic MPC in favor of continuous flow-matching paradigms, integrating world-action interactive feedback, dynamic activation-space steering, and discriminator-guided reinforcement learning to achieve autonomous unsupervised skill discovery.

## 2. Transitioning to Generative Action Denoising via Flow Matching

The inherent limitation of CEM is its reliance on sampling discrete low-level actions and evaluating their hypothetical latent futures. Generative sequence modeling reframes this by treating multi-step action generation as a continuous distribution matching problem, learning an ordinary differential equation (ODE) that smoothly transports a simple Gaussian prior into the complex data distribution of successful humanoid trajectories.

### 2.1. The World-Action Interactive Model (DAWN) Architecture

The structural paradigm proposed for "Phase 2 (Train Action Denoiser)" establishes a secondary network dedicated to trajectory generation [Image 1]. However, operating the world predictor and the action denoiser as isolated pipelines induces a predict-then-plan bottleneck. The DAWN (Denoising Actions and World iNteractive model) framework resolves this by enforcing action-contingent reciprocity.

In this integrated architecture, the system operates exclusively in the compact semantic latent space defined by the LeWM encoder. Let $z_t$ represent the multimodal latent state, and $a_{t:t+H}$ represent the proposed action horizon. Rather than rolling out a full future in pixel space, the system employs recursive interaction :

1.  The Action Denoiser initializes a noisy trajectory $a^{(K)}_{t:t+H} \sim \mathcal{N}(0, I)$, where $K$ is the maximum flow timestep [Image 1]
2.  The World Predictor rolls out a short explicit latent hypothesis $\hat{z}_{t+1:t+H}$ conditioned strictly on the current observation $z_t$ and the noisy action proposal.
3.  The Action Denoiser refines the trajectory to $a^{(k-1)}_{t:t+H}$, heavily conditioned on the predicted world rollout.

This interactive loop, where the denoised action updates the world prediction and vice versa, aligns the future world rollout and action generation into a self-consistent hypothesis. The denoiser weights remain shared across both the initial proposal and refinement roles, modulated entirely by distinct query and source embeddings.

### 2.2. Physics-Conditioned Flow and Modality-Specific Streams

To ensure the generative model inherently respects the physical boundaries of the Fourier GR-1, the flow matching architecture must be deeply conditioned on physical constraints [Image 1]. Drawing from the RLDX-1 architecture (Multi-Stream Action Transformer), heterogeneous modalities-such as raw RGB images, DINOv3 waypoints, tactile feedback matrices, and physics signals-are processed through modality-specific streams before cross-modal joint self-attention integration.

The conditioning vector for the flow transformer [Image 1] incorporates the proprioceptive and physics signals directly. Let $C_{phys}$ represent a dense tensor containing the controller gains and the $\mathcal{V}$-representation (vertices) of the velocity and force polytopes calculated via `pycapacity`. By conditioning the predicted velocity vector $v_{pred}$ on $C_{phys}$, the flow ODE is structurally biased to transport noise exclusively into the sub-manifold of physically executable joint trajectories.

Additionally, the OmniXtreme methodology provides a mechanism for decoupling general motor skill learning from sim-to-real physical refinement. An actuation-aware post-training phase can be applied to the action denoiser, utilizing extreme domain randomization of the URDF limits to ensure the flow-matched trajectories remain robust against the power-safety constraints and torque limits of physical hardware.

### 2.3. Egocentric 6DoF Generation and Tactile Alignment

For tasks requiring millimeter-precision during the pinch and lift phases, global visual representations are insufficient. Incorporating the EgoFlow paradigm, the global scene representation (extracted via DINOv3 and CLIP) is augmented with local, egocentric geometric features focusing specifically on bounding boxes around the end-effector and the target object. Using a bidirectional Mamba architecture flanking the core Flow Transformer [Image 1], the model captures long-horizon temporal dependencies while maintaining linear complexity, mapping point-cloud geometry to highly precise 6DoF object motion velocities.

Furthermore, the integration of Contrastive Language-Action Pretraining (CLAP) enforces an alignment between the visual latent space and the proprioceptive/tactile latent space. Because the cube pickup inherently demands tactile validation upon contact, a separate tactile evaluation stream acts as a critical checkpoint during the interactive denoising loop, allowing the model to abort and replan if the latent tactile prediction indicates a slippage event prior to the lift phase.

## 3. Hierarchical Trajectory Generation and Block Diffusion

Humanoid manipulation operates naturally across multiple temporal resolutions. A single, monolithic flow matching model attempting to map high-level task goals (e.g., "pick up the cube") directly to 10Hz joint torques across a 32-frame sequence suffers from severe gradient interference and optimization complexity.

### 3.1. Lifting Embodied World Models

To address this, the architecture adopts the "Lifting Embodied World Models" paradigm, which abstracts the generative process into two interconnected tiers. The high-level objective is formulated as generating sub-goal waypoints (e.g., the 2D image-space or 3D coordinate target for the leaf joints at the end of the approach or pre-grasp phases). In the proposed configuration [Image 1], the DINOv3 Projector acts as this high-level policy, mapping the current multi-view latents to a semantic waypoint.

The low-level Action Denoiser is then completely relieved of semantic reasoning; its sole objective is to generate the short-horizon continuous joint trajectory required to transition the robot from its current state to the proposed waypoint. This factorization significantly reduces the search space, allowing the flow matching model to operate effectively as a localized spline generator within the safe task-space polytope.

### 3.2. Block Discrete Denoising Diffusion (BD3)

This hierarchical separation naturally maps to the Block Diffusion framework. Block Diffusion interpolates between autoregressive (AR) language models and discrete denoising diffusion, defining an autoregressive probability distribution over blocks of variables, while the internal structure of each block is modeled via continuous or discrete diffusion.

In the context of the Fourier GR-1 pickup task, the high-level world predictor operates autoregressively, sequentially predicting the latent anchors for the four distinct phases (approach $\rightarrow$ pre-grasp $\rightarrow$ pinch $\rightarrow$ lift). Concurrently, the Action Denoiser executes parallel token sampling and continuous flow matching to generate the highly granular 8-frame micro-actions within each block. This enables the execution of the same high-level task through vast multitudes of low-level kinematic variations, exponentially increasing trajectory diversity and robustness.

### 3.3. Expanding the Exploration Manifold

A common failure mode in flow-matched behavior cloning is manifold collapse, where the generative model overfits to a narrow band of trajectories and fails catastrophically when introduced to minor OOD perturbations. The $\pi$-StepNFT (Step-wise Negative-Aware Fine-Tuning) and ComboStoc methodologies introduce mathematically rigorous countermeasures to this phenomenon.

$\pi$-StepNFT applies Stochastic Differential Equations (SDEs) to deliberately inject controlled noise during the flow trajectory inference, artificially expanding the exploration manifold. Similarly, ComboStoc replaces uniform scalar timesteps with vectorized, multidimensional stochastic schedules, allowing the model to independently noise and denoise disparate joints. This ensures that while the waist and arm may follow a deterministic path, the end-effector fingers explore a vast combinatorial space of approach angles, dramatically increasing the probability of a successful pinch on novel cube orientations.

## 4. Activation-Space Steering and Flow Inversion

While external action denoising provides robust trajectory generation, true generalist capabilities require the capacity to intrinsically alter the behavioral tendencies of the foundation model without the computational burden of parameter fine-tuning (e.g., LoRA). The deployment of Flow Matching can be extended beyond action generation to directly modulate the internal representations of the network.

### 4.1. UniSteer and Universal Conditional Velocity Fields

Activation-based control intervenes directly on the residual streams of frozen layers during inference. The UniSteer framework formulates this intervention not as a static additive vector, but as a text-guided or task-guided activation flow matching model.

Let $h^{(\ell)}$ represent the activation tensor at layer $\ell$ of the LeWM. Rather than learning a separate intervention for every possible sub-task, the system learns a universal conditional velocity field $v_\theta(h, t, c)$ in the high-dimensional activation space, where $c$ represents a human-readable condition, an extracted DINOv3 semantic tag, or a specific embodiment ID.

At inference time, when the robot transitions from the approach phase to the highly sensitive pinch phase, the model executes Flow Inversion :

1.  The standard unsteered activation $h_{source}$ is transported backward along the unconditional flow ODE to an intermediate noisy latent state $h_\tau$.
2.  This intermediate state is subsequently reconstructed forward under the specific target condition $c_{target}$ (e.g., "high-precision tactile contact").
3.  The resultant edited activation $\tilde{h}_{target}$ is injected back into the frozen LeWM, instantaneously altering the forward pass computations to reflect the new behavioral constraints.

This dynamic steering mechanism is structurally superior to LoRA, as it allows curved, multi-step, token-varying transport trajectories that adapt non-linearly to the current geometric state of the robot, overcoming the incomplete assumptions of linear activation geometry. The DeepVision-VLA architecture further refines this by applying sensitivity attenuation, ensuring that these steering vectors are primarily injected into the deeper layers of the transformer, which harbor the highest density of task-specific semantic routing.

| Steering Mechanism        | Operational Space   | Application Paradigm   | Computational Overhead                          |
|---------------------------|---------------------|------------------------|-------------------------------------------------|
| LoRA                      | Parameter Weights   | Offline Fine-Tuning    | High (Requires training distinct rank matrices) |
| Linear Probes             | Activation Space    | Static Addition        | Low (But suffers from geometric rigidity)       |
| UniSteer (Flow Inversion) | Activation Space    | Dynamic ODE Transport  | Moderate (Requires intermediate ODE solving)    |
| Flow Reversal Steering    | Action/Noise Space  | Behavioral Cloning     | Low (Executed strictly on trajectory outputs)   |

### 4.2. Flow Reversal Steering (FRS) for Sub-Optimal Trajectories

The newly acquired dataset presents a unique optimization opportunity: 1000 episodes contain sub-optimal trajectories that successfully navigate toward the cube but fail to execute the final pickup. Traditional supervised learning discards this data due to the lack of an ultimate success flag.

Flow Reversal Steering (FRS) provides a mathematical framework to extract the intrinsic spatial navigation intelligence embedded within these failures. Flow matching policies constitute a deterministic mapping between a Gaussian base distribution and the action space. FRS exploits this bijection by taking the suboptimal "reasonable" actions from the 1000-episode corpus and passing them through the learned flow policy in reverse .

By solving the reverse ODE, the system identifies the specific latent noise vectors $x_0$ that correspond to the failed physical actions. Once mapped into the noise space, these vectors are perturbed or projected toward nearby expert "modes"-areas of the latent space corresponding to the 500 successful pickups. This allows the system to train a noise-action policy via Noise-Space Behavioral Cloning (DSBC) or RL, effectively teaching the model how to correct its own near-misses without requiring massive volumes of new teleoperated expert data.

## 5. Scaling Reinforcement Learning on Generative Policies

With the generation of physically bounded trajectories established, the policy must be aligned to maximize the dense progress reward (inverse distance to the cube). However, applying Reinforcement Learning (RL) directly to diffusion or flow-matching models is notoriously difficult. Standard policy gradient methods require full Stochastic Differential Equation (SDE) rollouts at every optimization step to evaluate the reward, rendering training computationally intractable.

### 5.1. Reinforce Adjoint Matching (RAM)

To scale RL post-training for the Action Denoiser, the architecture implements Reinforce Adjoint Matching (RAM). The objective of RL on a generative model is to tilt the pretrained distribution toward high-reward samples while minimizing the Kullback-Leibler (KL) divergence from the original behavioral prior.

RAM translates this optimal control problem directly onto the velocity field $v_\theta$. Following the adjoint matching optimality condition, an optimal control perturbation can be formulated without backward adjoint sweeps or reward gradients. The algorithm executes as follows :

1.  A clean endpoint trajectory $x_0$ is drawn from the current on-policy model.
2.  The scalar reward $\mathcal{R}(x_0)$ is evaluated using the environment physics.
3.  The clean trajectory is analytically noised to an intermediate timestep $x_t$.
4.  The velocity field is updated via a simple regression loss that corrects the standard flow-matching target with the scalar reward multiplied by the difference between the noise and the data.

This elegant consistency loss achieves the reward-maximization capabilities of complex algorithms like Flow-GRPO in up to 50 times fewer training steps, fundamentally accelerating the alignment of the humanoid's motor control.

### 5.2. Discriminator-Guided RL (DRL) and Semantic Alignment

A critical vulnerability of RL in robotics is reward hacking; optimizing strictly for the inverse distance to the cube may result in the robot hovering endlessly over the target without executing the complex, multi-joint articulation required for the pinch . The model optimizes the mathematical proxy rather than the intended semantic behavior.

Discriminator-Guided RL (DRL) rectifies this structural mismatch. Rather than relying solely on the engineered reward, DRL trains a binary discriminator $D_\psi$ to separate the true 500 expert data samples $q$ from the base flow model's generative samples $p_{base}$. Crucially, this discriminator operates within a frozen, pretrained representation space (such as DINOv3) rather than raw pixels or joint angles, restricting the discriminator to focus strictly on perceptually and physically meaningful topological features.

The log-likelihood ratio between the data and the model is estimated via the discriminator's logit :

$$\hat{r}(x) = \text{logit}(D_\psi(\phi(x))) = -\log D_\psi(\phi(x_{real})) - \log(1 - D_\psi(\phi(x_{fake})))$$

This logit serves as the optimal dense reward for targeting the true data distribution. When this semantic reward is fed into the RAM algorithm, the Action Denoiser is forced to optimize for trajectories that not only achieve proximity to the cube but also perfectly mimic the structural flow and visual evolution of a human-piloted grasp.

### 5.3. BadWorld Adversarial Verification and EBM Fallbacks

To ensure the robustness of the trained policy, the BadWorld methodology is deployed during the RL phase. BadWorld utilizes label-free velocity attacks, introducing adversarial perturbations directly into the predicted velocity fields to actively hunt for vulnerabilities and edge cases within the policy's operational manifold.

When the system encounters highly uncertain states (e.g., navigating a novel obstacle introduced dynamically into the MuJoCo simulation), relying purely on the deterministic flow matching rollout may lead to failure. Here, Energy-Based Models (EBMs) serve as an interactive safeguard. The EBT-Policy (Energy-Based Transformers) framework parametrizes an implicit policy that computes a scalar energy landscape over all possible actions.

While flow matching provides high-speed trajectory generation in free space, the EBM dynamically evaluates the anomaly score of the proposed actions. If the energy crosses a critical threshold, the system shifts into a reactive mode, utilizing scaled Langevin Dynamics and Markov Chain Monte Carlo (MCMC) refinement to iteratively re-evaluate and correct the intermediate proposals before execution. This approach mirrors the "Just image Transformers" (JiT) philosophy, leveraging the low-dimensional manifold assumption to directly predict safe, clean data states when traversing highly corrupted or adversarial environment spaces.

## 6. Unsupervised Skill Discovery and On-Policy Self-Distillation

The introduction of 500 catastrophic failure episodes-where the humanoid underwent wild randomization-presents a unique frontier. Rather than discarding this data, it serves as the foundation for autonomous, unsupervised skill discovery. The goal is to evolve the GR-1 from a single-task specialist into a playful, curious agent capable of exploring its physical capabilities.

### 6.1. Privileged-Future OPSD (PF-OPSD)

World models inherently possess the capacity to simulate vast branching futures, but invoking this expansive rollout mechanism during real-time 10Hz control is impossible.

The architecture must distill this deep predictive capacity into a reactive policy. Privileged-Future On-Policy Self-Distillation (PF-OPSD) addresses this by intelligently transferring knowledge from an abstract-reasoning teacher to a concrete-reasoning student. During offline training, the "Teacher" network is granted access to the ground-truth future states $v^*$ from the exploration episodes, utilizing this privileged context to compute the mathematically optimal reaction sequence.

The "Student" network (the deployable Action Denoiser) is forced to minimize the divergence between its generated trajectory and the Teacher's optimal path, despite only possessing access to the current state observation $z_t$. This distills the profound foresight of the World Predictor directly into the base weights of the Action Denoiser, eliminating "Simulation Inertia"-the hesitation models exhibit when attempting to process complex future physics on the fly.

### 6.2. Diffusion-Tailored Suffix Conditioning (d-OPSD)

Standard OPSD frameworks inject privileged teacher information via left-to-right prefix conditioning, an autoregressive-centric design that fundamentally conflicts with the arbitrary-order, parallel-generation mechanics of flow matching and diffusion LLMs. To reconcile this, the architecture implements d-OPSD (Diffusion OPSD). This novel framework reformulates the self-teacher construction by utilizing the model's own self-generated physical outcomes as suffix conditioning . The student model learns iteratively from its "self future-experience" rather than relying on privileged prefixes. Crucially, d-OPSD shifts the divergence supervision from token-level cross-entropy to step-level velocity matching. By forcing the student's predicted flow velocity to match the teacher's velocity at every discrete ODE integration step, the model achieves a massive acceleration in sample efficiency-requiring only 10% of the optimization steps utilized by traditional RLVR baselines. This mechanism allows the robot to rapidly learn recovery behaviors from its 500 catastrophic failure episodes, turning random physical spasms into a library of generalized recovery skills.

### 6.3. Playful Agentic Robot Learning (RATs)

The integration of these distillation mechanisms culminates in the Robotics Agent Teams (RATs) paradigm. The GR-1 policy is endowed with a "play" stage, operating without explicit human instructions.

A Task Proposer agent navigates the DINOv3 semantic space to propose novel, learnable objectives (e.g., "push the cube," "flip the cube," "touch the table edge"). The Action Denoiser plans and executes these robot-code policies, while a Verification module evaluates the intermediate progress. Utilizing the Programmatic Skill Network (PSN) approach, successful executions are codified into executable symbolic programs with strict pre-conditions, control flows, and fault localizations, which are subsequently distilled into a persistent latent skill library. During downstream, zero-shot execution, if the robot encounters a novel obstacle, it simply retrieves the relevant latent skill vector from the library to condition the flow matching trajectory.

## 7. Future Anticipation and Hybrid Persistent Memory Architecture

Long-horizon exploration and complex skill chaining require an architectural capacity to maintain context far beyond the immediate observation window. A severe limitation of standard VLAs is memory fade; when an object like the tabletop cube is temporarily occluded by the humanoid's massive arm during the pre-grasp phase, standard transformers often hallucinate its disappearance, leading to terminal trajectory errors.

### 7.1. Latent Anticipation and World-Action Priors

Standard policies react to the present; advanced policies navigate the anticipated future. The World Pilot framework explicitly routes the priors of the World Action Model (WAM) into the decision chain. Latent Steering conditions the perception layer on a scene-evolution latent, while Action Steering supplies the anticipated trajectory as a motion prior to the denoiser.

This is augmented by the Future Forward Dynamics Causal Attention (FFDC) Verifier and models like HarmoWAM and Mantis. These models operate purely in the latent domain (predicting future conditions rather than raw pixels), continuously cross-referencing incoming sensory data against the LeWM's predicted trajectory. When executing the approach phase in free space, the system relies on long-horizon, high-speed action chunks. However, the moment contact forces are registered that deviate from the latent prediction (e.g., a glancing collision with the cube), the FFDC Verifier triggers an immediate halt, dropping the execution horizon to a single-step receding window for hyper-precise reactive adjustment.

### 7.2. The Hybrid Memory Triad (MemoryWAM, Echo-Memory, HM-World)

To support this continuous verification without overflowing GPU VRAM bounds, a highly specialized, multi-tiered memory architecture is required. Drawing from MemoryWAM , Echo-Memory , and HM-World , the system is designed to act simultaneously as a precise archivist for static environments and a vigilant tracker for dynamic, occluded subjects.

The Hybrid Memory Triad consists of three coupled mechanisms:

| Memory Tier   | Architectural Mechanism   | Operational Frequency   | Primary Function in Manipulation   |
|---------------|---------------------------|-------------------------|------------------------------------|

| Short-Term Context     | Sliding Observation Window     | High (10Hz continuous)          | Provides uncompressed, high-fidelity patches for immediate reactive physics control during contact.            |
|------------------------|--------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------|
| Event Boundary Anchors | Full Spatial Latents           | Triggered (e.g., contact onset) | Captures complete visual tokens at moments of high mnemonic salience to preserve task-onset goals.             |
| Long-Rang e Gist       | Relevance-Driv en HyDRA Tokens | Periodic / Compressed           | Condenses extensive historical rollouts into a minimal token footprint to maintain semantic object permanence. |

This structural decoupling reduces both time and space complexity during inference from $\mathcal{O}(N)$ to $\mathcal{O}(N/d)$ (where $d$ is the compression ratio), while fully preserving the persistent context required for non-Markovian decision-making. When the GR-1's hand occludes the cube, the spatiotemporal relevance-driven retrieval mechanism (HyDRA) selectively attends to the historical Gist Tokens , effectively remembering the cube's exact coordinate location and continuing the flow-matched pinch execution blindly.

Finally, borrowing from the "Language Models Need Sleep" paradigm, the agent utilizes offline computing cycles to consolidate these fragmented memories. Through an upward distillation process called Knowledge Seeding, short-term fragile interactions acquired during playful exploration are permanently etched into the deeper network weights, while a "Dreaming" phase uses RL to generate synthetic hallucinated curriculums, allowing the robot to mentally rehearse and refine its newly acquired skills without requiring physical hardware execution.

## 8. Synthesized Outcomes and Architectural Finalization

The transition from the heuristic boundaries of Latent MPC to a continuous, mathematically rigorous Flow Matching paradigm resolves the fundamental bottlenecks observed in the Fourier GR-1 tabletop manipulation task. The original CEM framework failed because it operated as a blind search engine attempting to navigate a deeply complex, contact-rich physical manifold.

The integration of DAWN's World-Action reciprocity ensures that the generative Action Denoiser proposes trajectories that are inextricably linked to the predicted physical consequences. By conditioning this flow matching ODE on the geometric vertices of the velocity and force polytopes provided by pycapacity , the model is strictly confined to generating physically executable kinematics, eliminating the OOD failures that plagued the original architecture.

Furthermore, leveraging the comprehensive 2000-episode dataset transitions the model from a narrow specialist to an adaptive generalist. Flow Reversal Steering (FRS) rescues the 1000 sub-optimal trajectories, converting them into valuable noise-space behavioral cloning targets. Concurrently, Discriminator-Guided RL (DRL) and Reinforce Adjoint Matching (RAM) align the generative policy to the highest semantic reward distributions without the computationally catastrophic burden of SDE rollouts. Fortified by dynamic activation-space steering (UniSteer) , fallback Energy-Based Models (EBT-Policy) , and a robust Hybrid Persistent Memory mechanism , the proposed architecture provides a complete, mathematically sound blueprint. This foundation empowers the humanoid not merely to execute a scripted cube pickup, but to engage in playful, unsupervised skill discovery (RATs, d-OPSD) , laying the groundwork for true autonomous physical intelligence.
