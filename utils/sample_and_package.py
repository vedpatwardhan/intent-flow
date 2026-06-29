import os
import zipfile


def sample_and_package():
    print("=== LatentFlow Raw Dataset Sampler & Packager ===")

    raw_data_dir = "latent-flow/data/raw"
    zip_filename = "dataset_samples.zip"

    if not os.path.exists(raw_data_dir):
        print(f"Error: Raw data directory '{raw_data_dir}' does not exist.")
        return

    # Find the first 3 episodes of Droid and CMU Stretch
    all_dirs = sorted(os.listdir(raw_data_dir))
    droid_eps = [d for d in all_dirs if d.startswith("bridge_ep")][:3]
    cmu_eps = [d for d in all_dirs if d.startswith("ego4d_ep")][:3]

    if not droid_eps or not cmu_eps:
        print(
            "Error: Could not find structured Droid or CMU episodes in the raw folder."
        )
        return

    print(f"Selected Droid episodes for sampling: {droid_eps}")
    print(f"Selected CMU episodes for sampling: {cmu_eps}")

    print(f"Packaging raw episodes into {zip_filename}...")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Package Droid Episodes
        for ep_name in droid_eps:
            ep_path = os.path.join(raw_data_dir, ep_name)
            frame_dir = os.path.join(ep_path, "frames")

            # Package all frames sequentially
            frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
            print(f"  Packaging Droid '{ep_name}' ({len(frames)} frames)...")
            for f in frames:
                zipf.write(
                    os.path.join(frame_dir, f),
                    arcname=f"raw/droid/{ep_name}/frames/{f}",
                )

            # Package state arrays
            for arr_file in ["actions.npy", "states.npy", "tactile.npy"]:
                fpath = os.path.join(ep_path, arr_file)
                if os.path.exists(fpath):
                    zipf.write(fpath, arcname=f"raw/droid/{ep_name}/{arr_file}")

        # Package CMU Stretch Episodes
        for ep_name in cmu_eps:
            ep_path = os.path.join(raw_data_dir, ep_name)
            frame_dir = os.path.join(ep_path, "frames")

            frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
            print(f"  Packaging CMU Stretch '{ep_name}' ({len(frames)} frames)...")
            for f in frames:
                zipf.write(
                    os.path.join(frame_dir, f), arcname=f"raw/cmu/{ep_name}/frames/{f}"
                )

            for arr_file in ["actions.npy", "states.npy", "tactile.npy"]:
                fpath = os.path.join(ep_path, arr_file)
                if os.path.exists(fpath):
                    zipf.write(fpath, arcname=f"raw/cmu/{ep_name}/{arr_file}")

    print(f"\n=== Completed! Raw sample package saved to '{zip_filename}' ===")


if __name__ == "__main__":
    sample_and_package()
