import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class PretrainingDataset(Dataset):
    """
    Loads preprocessed Stage 1 dataset .pt files.
    Slices each episode into a random temporal window of size window_size.
    Generates a boolean mask indicating which frames are masked for visual reconstruction.
    """

    def __init__(self, data_dir, window_size=32, mask_ratio=0.5, use_subset=False):
        # Auto-resolve "processed" subdirectory if it exists
        if os.path.exists(os.path.join(data_dir, "processed")):
            data_dir = os.path.join(data_dir, "processed")

        self.data_dir = data_dir
        self.window_size = window_size
        self.mask_ratio = mask_ratio

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Processed dataset directory not found at: {data_dir}"
            )

        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        if not self.files:
            raise ValueError(f"No processed .pt files found in {data_dir}")

        if use_subset:
            # Pick first 2 episodes from each of the three dataset groups if available
            subset = []
            try:
                droid_files = [
                    f for f in self.files if int(f.split("_")[1].split(".")[0]) <= 326
                ]
                cmu_files = [
                    f
                    for f in self.files
                    if 327 <= int(f.split("_")[1].split(".")[0]) <= 461
                ]
                odyssey_files = [
                    f for f in self.files if int(f.split("_")[1].split(".")[0]) >= 462
                ]

                subset.extend(droid_files[:2])
                subset.extend(cmu_files[:2])
                subset.extend(odyssey_files[:2])
            except Exception:
                pass

            if not subset:
                subset = self.files[:6]

            self.files = subset
            print(
                f"[Dataset] Running in diagnostic SUBSET mode. Loaded {len(self.files)} files: {self.files}"
            )
        else:
            print(
                f"[Dataset] Running in standard mode. Loaded {len(self.files)} files."
            )

        # Build window maps using a sliding window with stride 1 (matching le-probe)
        self.window_maps = []
        stride = 1
        for f_idx, fname in enumerate(self.files):
            file_path = os.path.join(self.data_dir, fname)
            # Load metadata (just checking sequence length)
            data = torch.load(file_path, map_location="cpu")
            total_len = data["vision"].shape[0]

            start_indices = list(range(0, total_len - self.window_size + 1, stride))
            if not start_indices:
                start_indices = [0]  # At least one window (to be padded if too short)

            for start_idx in start_indices:
                self.window_maps.append((fname, start_idx))

        print(
            f"[Dataset] Created {len(self.window_maps)} sliding windows (horizon: {self.window_size}) across {len(self.files)} episodes."
        )

    def __len__(self):
        return len(self.window_maps)

    def __getitem__(self, idx):
        fname, start_idx = self.window_maps[idx]
        file_path = os.path.join(self.data_dir, fname)
        data = torch.load(file_path, map_location="cpu")

        # 0 = Aloha, 1 = T-Rex, 2 = GR1 (Reserved for Stage 3)
        if "aloha" in file_path.lower():
            emb_id = 0
        elif "trex" in file_path.lower():
            emb_id = 1
        else:
            emb_id = 1  # Treat generic multi-DoF variant indices as default

        total_len = data["vision"].shape[0]
        end_idx = start_idx + self.window_size

        # Pad proprioception and actions along their last dimension to 58
        def pad_last_dim(t, target_dim=58):
            curr_dim = t.shape[-1]
            if curr_dim < target_dim:
                pad_shape = list(t.shape[:-1]) + [target_dim - curr_dim]
                pad = torch.zeros(pad_shape, dtype=t.dtype)
                return torch.cat([t, pad], dim=-1)
            return t

        # Pad vision along the views dimension (dim 1) to 4 views
        def pad_views(t, target_views=4):
            curr_views = t.shape[1]
            if curr_views < target_views:
                pad_shape = [t.shape[0], target_views - curr_views, t.shape[2]]
                pad = torch.zeros(pad_shape, dtype=t.dtype)
                return torch.cat([t, pad], dim=1)
            return t

        vision_padded = pad_views(data["vision"])
        proprio_padded = pad_last_dim(data["proprioception"])
        actions_padded = pad_last_dim(data["actions"])

        # Slice or pad to window_size
        if total_len >= end_idx:
            sliced_data = {
                "vision": vision_padded[start_idx:end_idx],
                "vggt": data["vggt"][start_idx:end_idx],
                "pointnext": data["pointnext"][start_idx:end_idx],
                "tactile": data["tactile"][start_idx:end_idx],
                "proprioception": proprio_padded[start_idx:end_idx],
                "actions": actions_padded[start_idx:end_idx],
                "text": data["text"],
                "embodiment_id": torch.tensor(emb_id, dtype=torch.long),
            }
        else:
            padding_len = self.window_size - total_len

            def pad_tensor(t, pad_val=0.0):
                pad_shape = [padding_len] + list(t.shape[1:])
                pad = torch.full(pad_shape, pad_val, dtype=t.dtype)
                return torch.cat([t, pad], dim=0)

            sliced_data = {
                "vision": pad_tensor(vision_padded),
                "vggt": pad_tensor(data["vggt"]),
                "pointnext": pad_tensor(data["pointnext"]),
                "tactile": pad_tensor(data["tactile"]),
                "proprioception": pad_tensor(proprio_padded),
                "actions": pad_tensor(actions_padded),
                "text": data["text"],
                "embodiment_id": torch.tensor(emb_id, dtype=torch.long),
            }

        # Generate random masking indices
        mask = torch.rand(self.window_size) < self.mask_ratio
        sliced_data["mask"] = mask

        return sliced_data


def get_dataloader(
    data_dir,
    seq_len=32,
    batch_size=32,
    use_subset=False,
    validation_split=0.0,
    num_workers=2,
):
    """
    Initializes PretrainingDataset and returns a PyTorch DataLoader (or a tuple of train/val DataLoaders if validation_split > 0).
    """
    dataset = PretrainingDataset(
        data_dir=data_dir, window_size=seq_len, mask_ratio=0.5, use_subset=use_subset
    )
    if validation_split > 0.0:
        train_size = int((1.0 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        if train_size == 0:
            train_size = len(dataset)
            val_size = 0
        if val_size > 0:
            train_set, val_set = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            return train_loader, val_loader
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            return loader, None
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
