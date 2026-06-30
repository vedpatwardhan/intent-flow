import os
import zipfile
import cv2
from PIL import Image


def compile_video(frame_dir, output_mp4_path, fps=15):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not frames:
        return False

    first_frame = Image.open(os.path.join(frame_dir, frames[0]))
    width, height = first_frame.size

    # Define vp09 (VP9) codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"vp09")
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))

    for f in frames:
        img_path = os.path.join(frame_dir, f)
        img = cv2.imread(img_path)
        out.write(img)
    out.release()
    return True


def sample_and_package(
    raw_data_dir="latent-flow/data/raw",
    processed_dir="latent-flow/data/processed",
    zip_filename="dataset_samples.zip",
):
    print("=== LatentFlow Raw & Processed Dataset Sampler & Validator ===")

    if not os.path.exists(raw_data_dir):
        print(f"Error: Raw data directory '{raw_data_dir}' does not exist.")
        return

    # Find the first 2 episodes of each category
    all_dirs = sorted(os.listdir(raw_data_dir))
    droid_eps = [d for d in all_dirs if d.startswith("bridge_ep")][:2]
    cmu_eps = [d for d in all_dirs if d.startswith("ego4d_ep")][:2]
    geom_eps = [d for d in all_dirs if d.startswith("geometry_ep")][:2]

    if not droid_eps or not cmu_eps or not geom_eps:
        print(
            "Error: Could not find structured Droid, CMU, or PointOdyssey episodes in raw data."
        )
        return

    # Temporary directory to store compiled MP4s
    temp_video_dir = "latent-flow/data/compiled_videos"
    os.makedirs(temp_video_dir, exist_ok=True)

    categories = [("droid", droid_eps), ("cmu", cmu_eps), ("geometry", geom_eps)]

    print(f"Compiling and packaging videos into {zip_filename}...")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for cat_name, ep_list in categories:
            print(f"\nProcessing category: {cat_name}")
            for ep_name in ep_list:
                ep_path = os.path.join(raw_data_dir, ep_name)
                frame_dir = os.path.join(ep_path, "frames")

                # 1. Compile all frames into a single MP4 video file
                video_filename = f"{cat_name}_{ep_name}.mp4"
                video_path = os.path.join(temp_video_dir, video_filename)

                print(f"  Compiling video for '{ep_name}'...")
                success = compile_video(frame_dir, video_path)

                if success:
                    # Save video into the zip archive
                    zipf.write(
                        video_path, arcname=f"videos/{cat_name}/{video_filename}"
                    )
                else:
                    print(f"  Warning: Failed to compile video for '{ep_name}'")

                # 2. Add raw state arrays (actions, states, tactile)
                for arr_file in ["actions.npy", "states.npy", "tactile.npy"]:
                    fpath = os.path.join(ep_path, arr_file)
                    if os.path.exists(fpath):
                        zipf.write(
                            fpath, arcname=f"arrays/{cat_name}/{ep_name}/{arr_file}"
                        )

    # 3. Processed Data Auditing & Packaging
    print("\n--- Auditing Processed (.pt) Representations ---")
    if os.path.exists(processed_dir):
        import torch

        proc_files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".pt")])
        if proc_files:
            print(
                f"Found {len(proc_files)} processed files. Auditing first file: {proc_files[0]}"
            )
            target_pt = os.path.join(processed_dir, proc_files[0])
            try:
                data = torch.load(target_pt, map_location="cpu")
                expected_keys = {
                    "vision": 384,
                    "text": 768,
                    "pointnext": 384,
                    "vggt": 768,
                    "tactile": (4, 4),
                    "proprioception": 24,
                    "actions": 12,
                }
                issues_found = 0
                for key, expected_dim in expected_keys.items():
                    if key not in data:
                        print(f"  [MISSING KEY] '{key}' is missing.")
                        issues_found += 1
                        continue
                    val = data[key]
                    shape_list = list(val.shape)
                    seq_len = data["vision"].shape[0]

                    if key == "text":
                        expected_shape = [1, 768]
                    elif key == "tactile":
                        expected_shape = [seq_len, 4, 4]
                    else:
                        expected_shape = [seq_len, expected_dim]

                    if shape_list != expected_shape:
                        print(
                            f"  [SHAPE WARNING] '{key}' shape {shape_list} != {expected_shape}"
                        )
                        issues_found += 1

                    mean_val = val.mean().item()
                    var_val = val.var().item()
                    is_dead = var_val == 0.0

                    status_str = "ACTIVE"
                    if is_dead:
                        if key in ["tactile", "proprioception"] and proc_files[
                            0
                        ].startswith("ego4d_"):
                            status_str = "ZERO-PAD (Expected for human demos)"
                        elif key == "tactile":
                            status_str = "ZERO-PAD (Expected for no-tactile setups)"
                        else:
                            status_str = "DEAD (Warning: Zero variance!)"
                            issues_found += 1

                    print(
                        f"  - {key:15s} | Shape: {str(shape_list):15s} | Mean: {mean_val:7.4f} | Var: {var_val:7.4f} | Status: {status_str}"
                    )

                # Write first processed PT file into the zip
                with zipfile.ZipFile(zip_filename, "a") as zipf:
                    zipf.write(target_pt, arcname=f"processed/{proc_files[0]}")
                print(
                    f"Successfully packaged '{proc_files[0]}' into ZIP processed/ folder."
                )
            except Exception as e:
                print(f"Failed to audit processed file: {e}")
        else:
            print("No processed files found to verify.")
    else:
        print(f"Processed directory '{processed_dir}' does not exist.")

    print(
        f"\n=== Completed! Package with 6 videos and audited processed file saved to '{zip_filename}' ==="
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Package raw dataset episodes and compile videos"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="latent-flow/data/raw",
        help="Path to the flat raw data directory on Colab",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="latent-flow/data/processed",
        help="Path to the processed data directory containing .pt files",
    )
    parser.add_argument(
        "--output_zip",
        type=str,
        default="dataset_samples.zip",
        help="Path/name of the generated ZIP output file",
    )
    args = parser.parse_args()
    sample_and_package(
        raw_data_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        zip_filename=args.output_zip,
    )
