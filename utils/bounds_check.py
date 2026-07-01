import os
import numpy as np
import argparse


def analyze_dataset_bounds(raw_dir="latent-flow/data/raw"):
    print(f"=== Analyzing LatentFlow Raw Dataset Bounds in: {raw_dir} ===")

    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist.")
        return

    episodes = sorted(
        [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    )
    print(f"Total episodes found: {len(episodes)}")

    # Group episodes by stream
    streams = {
        "droid (robot)": [e for e in episodes if e.startswith("bridge_ep")],
        "cmu (human)": [e for e in episodes if e.startswith("ego4d_ep")],
        "pointodyssey (geometry)": [e for e in episodes if e.startswith("geometry_ep")],
    }

    for stream_name, stream_eps in streams.items():
        print(f"\n--- Stream: {stream_name} (Found {len(stream_eps)} episodes) ---")
        if not stream_eps:
            print("No episodes found for this stream.")
            continue

        all_actions = []
        all_states = []
        all_tactile = []
        all_ptclouds = []

        for ep in stream_eps:
            ep_path = os.path.join(raw_dir, ep)

            action_file = os.path.join(ep_path, "actions.npy")
            if os.path.exists(action_file):
                all_actions.append(np.load(action_file))

            state_file = os.path.join(ep_path, "states.npy")
            if os.path.exists(state_file):
                all_states.append(np.load(state_file))

            tactile_file = os.path.join(ep_path, "tactile.npy")
            if os.path.exists(tactile_file):
                all_tactile.append(np.load(tactile_file))

            pt_file = os.path.join(ep_path, "point_clouds.npy")
            if os.path.exists(pt_file):
                all_ptclouds.append(np.load(pt_file))

        def print_stats(name, arrays):
            if not arrays:
                print(f"  No {name} files found.")
                return
            concat = np.concatenate(arrays, axis=0)
            print(f"  Combined {name} shape: {concat.shape}")

            # Reshape if multidimensional
            flat_shape = concat.shape
            if len(flat_shape) > 2:
                # E.g. tactile [N, 4, 4] or point clouds [N, 100, 4]
                dims = flat_shape[1:]
                print(f"  Checking {name} with dimensions {dims}...")
                # Flatten the trailing dimensions for bounds checking
                concat_flat = concat.reshape(concat.shape[0], -1)
            else:
                concat_flat = concat

            num_dim = concat_flat.shape[1]
            for d in range(num_dim):
                col = concat_flat[:, d]
                isnan = np.isnan(col).any()
                isinf = np.isinf(col).any()
                is_zero = (col == 0.0).all()
                min_val = col.min() if not isnan else float("nan")
                max_val = col.max() if not isnan else float("nan")
                mean_val = col.mean() if not isnan else float("nan")
                std_val = col.std() if not isnan else float("nan")

                print(
                    f"    Dim {d:02d}: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}, Std={std_val:.4f} | NaN/Inf: {isnan}/{isinf} | ConstantZero: {is_zero}"
                )

        print_stats("actions", all_actions)
        print_stats("states", all_states)
        print_stats("tactile", all_tactile)
        print_stats("point_clouds", all_ptclouds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze joint bounds and anomalies in prepared datasets"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="latent-flow/data/raw",
        help="Path to raw dataset directory",
    )
    args = parser.parse_args()
    analyze_dataset_bounds(args.raw_dir)
