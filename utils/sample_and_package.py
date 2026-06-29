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

    # Define mp4v codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))

    for f in frames:
        img_path = os.path.join(frame_dir, f)
        img = cv2.imread(img_path)
        out.write(img)
    out.release()
    return True


def sample_and_package(
    raw_data_dir="latent-flow/data/raw", zip_filename="dataset_samples.zip"
):
    print("=== LatentFlow Raw Dataset Video Compiler & Packager ===")

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

    print(
        f"\n=== Completed! Package with 6 compiled videos saved to '{zip_filename}' ==="
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
        "--output_zip",
        type=str,
        default="dataset_samples.zip",
        help="Path/name of the generated ZIP output file",
    )
    args = parser.parse_args()
    sample_and_package(raw_data_dir=args.raw_dir, zip_filename=args.output_zip)
