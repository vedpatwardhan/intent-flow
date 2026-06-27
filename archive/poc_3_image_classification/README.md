# PoC 3: Image Classification with Flow-Guided Output Probabilities

This Proof-of-Concept (PoC) evaluates the performance of Flow-Guided training on a static supervised classification task using the MNIST dataset.

## Objective
To determine if guiding a classifier's output probabilities along a straight geodesic path (from initial random predictions to clean target one-hot class distributions) accelerates convergence and straightens training trajectories.

## Setup & Architectures
*   **Dataset**: MNIST digits (subsampled to 2,000 images for fast CPU execution).
*   **Target Model**: 
    *   *Default*: A simple feedforward neural network (`784 -> 128 -> 64 -> 10` Tanh MLP, 109k parameters).
    *   *Lightweight*: A single linear classification layer (`784 -> 10`, 7.8k parameters).
*   **Flow Matching Model**: A tiny MLP mapping inputs of current probability, time, and image context ($p_t, t, x_{\text{cond}}$) to velocity vectors (ranging from 618 to 5k parameters).
*   **Conditioning**: A fixed random projection matrix compresses the 784D images to a low-dimensional context (16D/128D) to keep the flow model small ($<1.5\%$ of target model parameters).

## Run Instructions
Run the data visualization script to see raw digits and projection manifolds:
```bash
.venv/bin/python flow-train/poc_3_image_classification/visualize_data.py
```

Run the main training script to execute the A/B comparative benchmark:
```bash
.venv/bin/python flow-train/poc_3_image_classification/flow_matching.py
```

## Key Findings & Takeaways
1.  **Redundancy on Simplex**: In probability space, standard unguided Cross-Entropy training naturally takes almost perfectly straight trajectories from uniform initialization to target class vertices ($\text{Straightness} \approx 0.98$). Flow-guiding the outputs is redundant.
2.  **Noise Memorization**: Defining the flow matching path starting from a random initialization ($p_{\text{init}}$) forces the network to learn a mapping that memorizes unstructured, image-specific initial noise signatures. This severely slows down convergence and bends the training path ($\text{Straightness} \approx 0.70$).
3.  **Speed Limits**: Tying the flow matching time $t$ to a fixed epoch schedule imposes an artificial "speed limit" on convergence, preventing the model from converging faster than the linear schedule.
4.  **Rank Verification**: Although convergence is slower, Flow-Guided parameter updates ($\Delta W$) quantifiably exhibit a faster singular value decay (lower rank) than baseline unguided updates, supporting the parameter rank constraint hypothesis.

## Conclusion
Output-space flow matching from scratch is not viable for standard image classification. The flow matching paradigm is instead highly suited to **Fine-Tuning (SFT)** from a pre-trained base (where starting states are structured rather than random noise) or directly to **parameter-space weight flows (like LoRA path straightening)**.
