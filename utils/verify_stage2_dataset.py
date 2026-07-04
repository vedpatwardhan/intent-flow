import os
import torch
import numpy as np
from tqdm import tqdm


def verify_stage2_dataset(processed_dir):
    print("=== Running Stage 2 Dataset Global Verification ===")
    if not os.path.exists(processed_dir):
        print(f"Error: Processed directory not found at {processed_dir}")
        return

    pt_files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".pt")])
    if not pt_files:
        print(f"Error: No .pt files found in {processed_dir}")
        return

    print(f"Found {len(pt_files)} processed episodes. Verifying integrity...")

    corrupted = 0
    aloha_count = 0
    trex_count = 0
    total_frames = 0

    for fname in tqdm(pt_files, desc="Checking files"):
        path = os.path.join(processed_dir, fname)
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[ERROR] Failed to load {fname}: {e}")
            corrupted += 1
            continue

        # Check required keys
        required_keys = [
            "vision",
            "text",
            "pointnext",
            "vggt",
            "tactile",
            "proprioception",
            "actions",
        ]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            print(f"[ERROR] {fname} is missing keys: {missing_keys}")
            corrupted += 1
            continue

        # Check for NaNs or Infs in any tensor
        has_nan_or_inf = False
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"[ERROR] {fname} contains NaN or Inf in key: {k}")
                    has_nan_or_inf = True
                    break
        if has_nan_or_inf:
            corrupted += 1
            continue

        # Verify shapes
        seq_len = data["vision"].shape[0]
        total_frames += seq_len

        # Count dataset mixture
        if "aloha" in fname.lower():
            aloha_count += 1
        elif "trex" in fname.lower():
            trex_count += 1

    print("\n--- Global Sanity Check Summary ---")
    print(f"Total processed files scanned: {len(pt_files)}")
    print(f"Successfully verified: {len(pt_files) - corrupted}")
    print(f"Corrupted / Failed files: {corrupted}")
    print(f"Total SFT dataset frames: {total_frames}")
    print(f"Mixture: ALOHA ({aloha_count} episodes), T-REX ({trex_count} episodes)")
    print("-----------------------------------\n")

    # Perform Deep Inspection on samples
    inspect_samples(processed_dir, pt_files)


def inspect_samples(processed_dir, pt_files):
    print("=== Running Stage 2 Episode Sampling & Inspection ===")

    # Locate sample files
    aloha_samples = [f for f in pt_files if "aloha" in f.lower()][:2]
    trex_samples = [f for f in pt_files if "trex" in f.lower()][:2]

    samples_to_inspect = aloha_samples + trex_samples
    if not samples_to_inspect:
        print("No sample episodes found to inspect.")
        return

    for fname in samples_to_inspect:
        print(f"\n[INSPECTING] File: {fname}")
        path = os.path.join(processed_dir, fname)
        data = torch.load(path, map_location="cpu")

        # 1. Episode specs
        seq_len = data["vision"].shape[0]
        num_views = data["vision"].shape[1] if data["vision"].dim() > 2 else 1
        print(f"  * Length (Steps): {seq_len}")
        print(f"  * Camera Views Count: {num_views}")

        # 2. Vision Features
        print(
            f"  * Vision Tensor Shape: {list(data['vision'].shape)} (Expected: [Seq, Views, 384])"
        )

        # 3. Actions
        act = data["actions"]
        print(f"  * Actions Shape: {list(act.shape)} (Expected: [Seq, ActionDim])")
        print(
            f"  * Actions Stats - Min: {act.min().item():.4f}, Max: {act.max().item():.4f}, Mean: {act.mean().item():.4f}, Std: {act.std().item():.4f}"
        )

        # 4. Proprioception / State
        state = data["proprioception"]
        print(f"  * State Shape: {list(state.shape)} (Expected: [Seq, StateDim])")
        print(
            f"  * State Stats   - Min: {state.min().item():.4f}, Max: {state.max().item():.4f}, Mean: {state.mean().item():.4f}, Std: {state.std().item():.4f}"
        )

        # 5. Tactile force
        tac = data["tactile"]
        print(f"  * Tactile Shape: {list(tac.shape)} (Expected: [Seq, 4, 4])")
        # Check if tactile force contains values or is zeroed out
        is_zero = torch.all(tac == 0).item()
        if is_zero:
            print("  * Tactile Data: [ZEROED] (Expected for ALOHA)")
        else:
            print(
                f"  * Tactile Stats - Min: {tac.min().item():.4f}, Max: {tac.max().item():.4f}, Mean: {tac.mean().item():.4f}"
            )

        # 6. PointNeXt Geometric tokens
        pn = data["pointnext"]
        print(
            f"  * PointNeXt Shape: {list(pn.shape)} (Expected: [Seq, 32, 384] or matching adapter input)"
        )

        # 7. CLIP Text prompt
        text = data["text"]
        print(f"  * Text Shape: {list(text.shape)} (Expected: [1, 768])")

    print("\n=== Episode Inspection Complete ===")


if __name__ == "__main__":
    processed_dir = "latent-flow/data/processed/sft"
    verify_stage2_dataset(processed_dir)
