import base64
import io
import os
import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFilter


def decode_base64_image(base64_str: str) -> np.ndarray:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return np.array(img)


def save_stage3_debug_plots(payload, obs_dict: dict, goal_images: dict):
    """
    Decodes multi-view frames, creates semi-transparent overlays for annotations,
    and saves debug_stage3_step.png and debug_goal_states.png.
    """
    camera_names = [
        "world_center",
        "world_top",
        "world_left",
        "world_right",
        "world_wrist",
    ]
    decoded_images = {}
    for cam in camera_names:
        if cam in payload.frames:
            try:
                decoded_images[cam] = Image.fromarray(
                    decode_base64_image(payload.frames[cam])
                )
            except Exception as e:
                print(f"Error decoding {cam} in save_stage3_debug_plots: {e}")
                decoded_images[cam] = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            decoded_images[cam] = Image.new("RGB", (224, 224), (0, 0, 0))

    # 1. Dynamic subplots plot
    N = len(payload.ui_annotations) if payload.ui_annotations else 0
    total_plots = 5 + N
    cols = 3
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if total_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Turn off axis on all subplots initially
    for ax in axes:
        ax.axis("off")

    # Plot base 5 camera views
    for idx, cam in enumerate(camera_names):
        axes[idx].imshow(decoded_images[cam])
        axes[idx].set_title(cam)

    # Plot each annotated view overlay
    for idx, (view_name, view_annos) in enumerate(payload.ui_annotations.items()):
        if view_name not in decoded_images:
            continue

        # Subplot index starts at 5
        plot_idx = 5 + idx
        if plot_idx >= len(axes):
            break

        try:
            overlay_img = decoded_images[view_name].copy().convert("RGBA")
            crops = view_annos.get("crops", [])
            segments = view_annos.get("segments", [])
            vectors = view_annos.get("vectors", [])
            img_w, img_h = decoded_images[view_name].size

            active_annos = crops + segments
            if len(active_annos) >= 2:
                p0_anno = active_annos[0]
                p1_anno = active_annos[1]

                view_features_dict = obs_dict.get("view_features", {})
                task_isolated = view_features_dict.get(view_name, {}).get(
                    "task_isolated_features", {}
                )
                sam_mask_224 = task_isolated.get("sam_mask_224", None)

                def get_coords(anno):
                    scale_x = img_w / 224.0
                    scale_y = img_h / 224.0
                    is_crop_type = "width" in anno
                    if is_crop_type:
                        x = int(anno["x"] * scale_x)
                        y = int(anno["y"] * scale_y)
                        w = int(anno["width"] * scale_x)
                        h = int(anno["height"] * scale_y)
                        mask = Image.new("L", (w, h), 255)
                    else:
                        if sam_mask_224 is not None and np.sum(sam_mask_224) > 0:
                            try:
                                mask_np_224 = (
                                    np.array(sam_mask_224)
                                    if isinstance(sam_mask_224, torch.Tensor)
                                    else sam_mask_224
                                )
                                mask_uint8 = (mask_np_224 > 0).astype(np.uint8) * 255
                                num_labels, labels = cv2.connectedComponents(mask_uint8)

                                cx_scaled = min(223, max(0, int(anno["x"])))
                                cy_scaled = min(223, max(0, int(anno["y"])))
                                lbl = labels[cy_scaled, cx_scaled]

                                if lbl == 0:
                                    window = labels[
                                        max(0, cy_scaled - 5) : min(224, cy_scaled + 6),
                                        max(0, cx_scaled - 5) : min(224, cx_scaled + 6),
                                    ]
                                    non_zero = window[window > 0]
                                    if len(non_zero) > 0:
                                        lbl = non_zero[0]

                                if lbl > 0:
                                    segment_mask_224 = (labels == lbl).astype(
                                        np.float32
                                    )
                                    mask_pil = Image.fromarray(
                                        (segment_mask_224 * 255).astype(np.uint8)
                                    )
                                    mask_resized = mask_pil.resize(
                                        (img_w, img_h), Image.NEAREST
                                    )
                                    mask_np = np.array(mask_resized)

                                    indices = np.argwhere(mask_np > 0)
                                    y_min, x_min = indices.min(axis=0)
                                    y_max, x_max = indices.max(axis=0)

                                    x = int(x_min)
                                    y = int(y_min)
                                    w = int(x_max - x_min + 1)
                                    h = int(y_max - y_min + 1)
                                    mask = mask_resized.crop((x, y, x + w, y + h))
                                else:
                                    raise ValueError(
                                        "No matching component label found"
                                    )
                            except Exception as e:
                                print(
                                    f"Fallback to circle due to components error: {e}"
                                )
                                cx = int(anno["x"] * scale_x)
                                cy = int(anno["y"] * scale_y)
                                r = int(25 * scale_x)
                                x = max(0, cx - r)
                                y = max(0, cy - r)
                                w = min(img_w - x, 2 * r)
                                h = min(img_h - y, 2 * r)
                                mask = Image.new("L", (w, h), 0)
                                draw = ImageDraw.Draw(mask)
                                draw.ellipse(
                                    (cx - r - x, cy - r - y, cx + r - x, cy + r - y),
                                    fill=255,
                                )
                        else:
                            cx = int(anno["x"] * scale_x)
                            cy = int(anno["y"] * scale_y)
                            r = int(25 * scale_x)
                            x = max(0, cx - r)
                            y = max(0, cy - r)
                            w = min(img_w - x, 2 * r)
                            h = min(img_h - y, 2 * r)
                            mask = Image.new("L", (w, h), 0)
                            draw = ImageDraw.Draw(mask)
                            draw.ellipse(
                                (cx - r - x, cy - r - y, cx + r - x, cy + r - y),
                                fill=255,
                            )
                    return x, y, w, h, mask

                x1, y1, w1, h1, mask1 = get_coords(p0_anno)
                x2, y2, w2, h2, mask2 = get_coords(p1_anno)

                # Swap based on vector direction
                if vectors and len(vectors) > 0:
                    vec = vectors[0]
                    scale_x = img_w / 224.0
                    scale_y = img_h / 224.0
                    start_x = vec["start"][0] * scale_x
                    start_y = vec["start"][1] * scale_y
                    ctr0_x = x1 + w1 / 2.0
                    ctr0_y = y1 + h1 / 2.0
                    ctr1_x = x2 + w2 / 2.0
                    ctr1_y = y2 + h2 / 2.0
                    d0_start = (ctr0_x - start_x) ** 2 + (ctr0_y - start_y) ** 2
                    d1_start = (ctr1_x - start_x) ** 2 + (ctr1_y - start_y) ** 2
                    if d1_start < d0_start:
                        x1, y1, w1, h1, mask1, x2, y2, w2, h2, mask2 = (
                            x2,
                            y2,
                            w2,
                            h2,
                            mask2,
                            x1,
                            y1,
                            w1,
                            h1,
                            mask1,
                        )

                # Paste overlays
                # Patch 1 (moving): semi-transparent cyan
                cyan_color = Image.new("RGBA", (w1, h1), (0, 255, 255, 80))
                overlay_img.paste(cyan_color, (x1, y1), mask1)

                # Patch 2 (fixed): semi-transparent magenta
                magenta_color = Image.new("RGBA", (w2, h2), (255, 0, 255, 80))
                overlay_img.paste(magenta_color, (x2, y2), mask2)
        except Exception as e:
            print(f"Error drawing overlays for {view_name}: {e}")
            overlay_img = decoded_images[view_name]

        axes[plot_idx].imshow(overlay_img)
        axes[plot_idx].set_title(f"Annotations & Patches ({view_name})")
        axes[plot_idx].axis("on")

        img_w, img_h = decoded_images[view_name].size
        scale_x = img_w / 224.0
        scale_y = img_h / 224.0

        crops = view_annos.get("crops", [])
        for crop in crops:
            rect = patches.Rectangle(
                (crop["x"] * scale_x, crop["y"] * scale_y),
                crop["width"] * scale_x,
                crop["height"] * scale_y,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axes[plot_idx].add_patch(rect)

        segments = view_annos.get("segments", [])
        for seg in segments:
            x = seg.get("x", 0) * scale_x
            y = seg.get("y", 0) * scale_y
            axes[plot_idx].plot(
                x, y, marker="x", color="red", markersize=8, markeredgewidth=2
            )

        vectors = view_annos.get("vectors", [])
        for vec in vectors:
            start_x = vec["start"][0] * scale_x
            start_y = vec["start"][1] * scale_y
            end_x = vec["end"][0] * scale_x
            end_y = vec["end"][1] * scale_y
            axes[plot_idx].annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(
                    arrowstyle="->", color="cyan", lw=2.5, mutation_scale=15
                ),
            )

    plt.tight_layout()
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "debug_stage3_step.png"
    )
    plt.savefig(output_path)
    plt.close()

    # 2. goal states plot per view
    for view_name, goal_data in goal_images.items():
        images = goal_data.get("goal_frames", [])
        if not images:
            continue
        fig_goals, axes_goals = plt.subplots(2, 2, figsize=(10, 10))
        axes_goals = axes_goals.flatten()
        names = ["left", "right", "top", "bottom"]
        for idx, (name, img) in enumerate(zip(names, images)):
            axes_goals[idx].imshow(img)
            axes_goals[idx].set_title(
                f"Goal State ({view_name}): Patch 1 on {name.capitalize()} of Patch 2"
            )
            axes_goals[idx].axis("off")
        plt.tight_layout()
        goals_output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            f"debug_goal_states_{view_name}.png",
        )
        plt.savefig(goals_output_path)
        plt.close()


def save_stage3_goal_features_plots(
    clean_image_pil, ui_annotations, obs_dict, view_name="world_center"
):
    """
    Saves a 4-panel diagnostic plot comparing:
    1. Original Clean Base Image (I_0)
    2. UI Drawing Overlay (Segments + Drawn Intent Arrow)
    3. Transformed DINOv3 Feature Map Overlay (Latent Arm Bridge)
    4. Transformed VGGT Motion Trajectory Field Overlay (Forward Velocity Vectors)
    """
    try:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        img_w, img_h = clean_image_pil.size
        scale_x = img_w / 224.0
        scale_y = img_h / 224.0

        # Retrieve annotations specifically for this view
        view_annos = (
            ui_annotations.get(view_name, {})
            if isinstance(ui_annotations.get(view_name), dict)
            else ui_annotations
        )

        # --- Panel 1: Original Clean Base Image ---
        axes[0].imshow(clean_image_pil)
        axes[0].set_title(f"1. Original Clean Base Image ({view_name})", fontsize=10)
        axes[0].axis("off")

        # --- Panel 2: UI Drawing Overlay (Segments & Arrow) ---
        overlay_img = clean_image_pil.copy().convert("RGBA")
        draw = ImageDraw.Draw(overlay_img)

        crops = view_annos.get("crops", [])
        for crop in crops:
            cx1 = crop["x"] * scale_x
            cy1 = crop["y"] * scale_y
            cx2 = (crop["x"] + crop["width"]) * scale_x
            cy2 = (crop["y"] + crop["height"]) * scale_y
            draw.rectangle([cx1, cy1, cx2, cy2], outline="lime", width=3)

        segments = view_annos.get("segments", [])
        for seg in segments:
            sx = seg["x"] * scale_x
            sy = seg["y"] * scale_y
            draw.ellipse(
                [sx - 5, sy - 5, sx + 5, sy + 5], fill="red", outline="white", width=2
            )

        vectors = view_annos.get("vectors", [])
        for vec in vectors:
            sx = vec["start"][0] * scale_x
            sy = vec["start"][1] * scale_y
            ex = vec["end"][0] * scale_x
            ey = vec["end"][1] * scale_y
            draw.line([(sx, sy), (ex, ey)], fill="cyan", width=4)

        axes[1].imshow(overlay_img)
        axes[1].set_title("2. UI Drawing Overlay (Segments & Arrow)", fontsize=10)
        axes[1].axis("off")

        # --- Panel 3: Transformed DINOv3 Feature Map Overlay (Full attention + Arm Bridge) ---
        dino_map = obs_dict[view_name]["vision"].squeeze(0)[:196].view(14, 14)
        dino_map = dino_map.detach().cpu().numpy()

        axes[2].imshow(clean_image_pil)
        axes[2].imshow(
            dino_map,
            cmap="jet",
            alpha=0.45,
            extent=[0, img_w, img_h, 0],
            interpolation="bilinear",
        )
        axes[2].set_title("3. Transformed DINOv3 (Latent Arm Bridge)", fontsize=10)
        axes[2].axis("off")

        # --- Panel 4: Transformed VGGT Motion Trajectory Field Overlay (Full Flow) ---
        vggt_tensor = obs_dict[view_name]["vggt"].squeeze(0)
        vggt_map = vggt_tensor.detach().cpu().numpy()

        axes[3].imshow(clean_image_pil)
        axes[3].imshow(
            vggt_map,
            cmap="jet",
            alpha=0.45,
            extent=[0, img_w, img_h, 0],
            interpolation="bilinear",
        )
        axes[3].set_title("4. Transformed VGGT (Trajectory Field)", fontsize=10)
        axes[3].axis("off")

        # --- Panel 5: Transformed CLIP (Segment Transfer) ---
        clip_sim = obs_dict[view_name]["text"].squeeze(0)[:196].view(14, 14)
        clip_sim = clip_sim.detach().cpu().numpy()

        axes[4].imshow(clean_image_pil)
        axes[4].imshow(
            clip_sim,
            cmap="jet",
            alpha=0.45,
            extent=[0, img_w, img_h, 0],
            interpolation="bilinear",
        )
        axes[4].set_title("5. Transformed CLIP (Segment Transfer)", fontsize=10)
        axes[4].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            f"debug_goal_features_{view_name}.png",
        )
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error saving stage3 goal feature comparison plot: {e}")


def save_stage3_oracle_goal_plots(
    oracle_payload, oracle_obs_dict, view_name="world_center"
):
    """
    Saves a 4-panel diagnostic PNG plot comparing:
    1. Initial Environment Starting Posture (I_0)
    2. Ground-Truth Oracle Target Goal Posture (I_goal)
    3. Target DINOv3 Feature Map Overlay (I_goal_DINO)
    4. Target VGGT Motion Trajectory Field Overlay (I_goal_VGGT)
    """
    try:
        # Extract starting frame and goal frame from oracle_payload
        history_frames = [frames[view_name] for frames in oracle_payload.history_frames]
        start_frame_str = history_frames[-4]
        oracle_frame_str = history_frames[-1]
        start_np = decode_base64_image(start_frame_str)
        oracle_np = decode_base64_image(oracle_frame_str)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        img_h, img_w, _ = start_np.shape

        # --- Panel 1: Initial Environment Starting Posture ---
        axes[0].imshow(start_np)
        axes[0].set_title(f"1. Initial Start Posture ({view_name})", fontsize=10)
        axes[0].axis("off")

        # --- Panel 2: Ground-Truth Oracle Target Goal Posture ---
        axes[1].imshow(oracle_np)
        axes[1].set_title(
            f"2. Ground-Truth Oracle Goal Snapshot ({view_name})", fontsize=10
        )
        axes[1].axis("off")

        # --- Panel 3: Target DINOv3 Feature Map Overlay ---
        dino_map = oracle_obs_dict[view_name]["vision"].squeeze(0)[:196].view(14, 14)
        dino_map = dino_map.detach().cpu().numpy()

        axes[2].imshow(oracle_np)
        axes[2].imshow(
            dino_map,
            cmap="jet",
            alpha=0.45,
            extent=[0, img_w, img_h, 0],
            interpolation="bilinear",
        )
        axes[2].set_title("3. Oracle Goal DINOv3 Feature Map", fontsize=10)
        axes[2].axis("off")

        # --- Panel 4: Target VGGT Motion Trajectory Field Overlay ---
        vggt_tensor = oracle_obs_dict[view_name]["vggt"].squeeze(0)
        vggt_map = vggt_tensor.detach().cpu().numpy()

        axes[3].imshow(oracle_np)
        axes[3].imshow(
            vggt_map,
            cmap="jet",
            alpha=0.45,
            extent=[0, img_w, img_h, 0],
            interpolation="bilinear",
        )
        axes[3].set_title("4. Oracle Goal VGGT Trajectory Field", fontsize=10)
        axes[3].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            f"debug_oracle_goal_{view_name}.png",
        )
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"📸 Saved 4-Panel Oracle Goal diagnostic plot to: {output_path}")
    except Exception as e:
        print(f"Error saving stage3 oracle goal diagnostic plot: {e}")


def save_stage3_calibrate_plots(
    trans_payload, obs_dict_next, view_name: str = "world_center", track_idx: int = 0
):
    """
    Saves a 4-panel diagnostic PNG plot verifying calibration transition observation features:
    1. Transition Start Frame (history_frames[0])
    2. Transition Outcome Frame (history_frames[-1])
    3. Outcome Frame DINOv3 Feature Map Overlay
    4. 4-Frame Accumulated VGGT Motion Trajectory Field Overlay
    """
    try:
        history_frames = trans_payload.next_obs.history_frames
        start_frame_str = history_frames[0][view_name]
        outcome_frame_str = history_frames[-1][view_name]
        start_np = decode_base64_image(start_frame_str)
        outcome_np = decode_base64_image(outcome_frame_str)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        img_h, img_w, _ = outcome_np.shape

        # --- Panel 1: Transition Start Posture ---
        axes[0].imshow(start_np)
        axes[0].set_title(
            f"1. Track {track_idx} Start Posture ({view_name})", fontsize=10
        )
        axes[0].axis("off")

        # --- Panel 2: Transition Outcome Posture ---
        axes[1].imshow(outcome_np)
        axes[1].set_title(
            f"2. Track {track_idx} Outcome Posture ({view_name})", fontsize=10
        )
        axes[1].axis("off")

        # --- Panel 3: Outcome DINOv3 Feature Map Overlay ---
        dino_map = obs_dict_next[view_name]["vision"].squeeze(0)[:196].view(14, 14)
        dino_map = dino_map.detach().cpu().numpy()

        axes[2].imshow(outcome_np)
        axes[2].imshow(
            dino_map,
            cmap="jet",
            alpha=0.45,
            extent=[0, img_w, img_h, 0],
            interpolation="bilinear",
        )
        axes[2].set_title(f"3. Track {track_idx} DINOv3 Map", fontsize=10)
        axes[2].axis("off")

        # --- Panel 4: 4-Frame Accumulated VGGT Motion Field Overlay ---
        vggt_tensor = obs_dict_next[view_name]["vggt"].squeeze(0)
        vggt_map = vggt_tensor.detach().cpu().numpy()

        axes[3].imshow(outcome_np)
        axes[3].imshow(
            vggt_map,
            cmap="jet",
            alpha=0.45,
            extent=[0, img_w, img_h, 0],
            interpolation="bilinear",
        )
        axes[3].set_title(f"4. Track {track_idx} VGGT Motion Field", fontsize=10)
        axes[3].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            f"debug_calibrate_track_{track_idx}_{view_name}.png",
        )
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"📸 Saved 4-Panel Calibrate Track {track_idx} plot to: {output_path}")
    except Exception as e:
        print(f"Error saving stage3 calibrate diagnostic plot: {e}")
