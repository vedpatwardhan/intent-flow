import os
import torch
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    """
    Loads preprocessed Stage 1 dataset .pt files.
    Slices each episode into a random temporal window of size window_size.
    Generates a boolean mask indicating which frames are masked for visual reconstruction.
    """

    def __init__(self, data_dir, window_size=32, mask_ratio=0.5, use_subset=False):
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
            # Restrict to first 5 files for dry-run verification
            self.files = self.files[:5]
            print(
                f"[Dataset] Running in diagnostic SUBSET mode. Loaded {len(self.files)} files."
            )
        else:
            print(
                f"[Dataset] Running in standard mode. Loaded {len(self.files)} files."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path, map_location="cpu")

        # Determine sequence length
        total_len = data["vision"].shape[0]

        # 1. Slice or pad to window_size
        if total_len > self.window_size:
            # Random starting index for window slicing
            start_idx = torch.randint(0, total_len - self.window_size + 1, (1,)).item()
            end_idx = start_idx + self.window_size

            sliced_data = {
                "vision": data["vision"][start_idx:end_idx],
                "vggt": data["vggt"][start_idx:end_idx],
                "pointnext": data["pointnext"][start_idx:end_idx],
                "tactile": data["tactile"][start_idx:end_idx],
                "proprioception": data["proprioception"][start_idx:end_idx],
                "actions": data["actions"][start_idx:end_idx],
                "text": data[
                    "text"
                ],  # text instruction is shape [1, 768] (sequence-independent)
            }
        else:
            # Pad sequences shorter than window_size
            padding_len = self.window_size - total_len

            def pad_tensor(t, pad_val=0.0):
                pad_shape = [padding_len] + list(t.shape[1:])
                pad = torch.full(pad_shape, pad_val, dtype=t.dtype)
                return torch.cat([t, pad], dim=0)

            sliced_data = {
                "vision": pad_tensor(data["vision"]),
                "vggt": pad_tensor(data["vggt"]),
                "pointnext": pad_tensor(data["pointnext"]),
                "tactile": pad_tensor(data["tactile"]),
                "proprioception": pad_tensor(data["proprioception"]),
                "actions": pad_tensor(data["actions"]),
                "text": data["text"],
            }

        # 2. Generate random masking indices
        # Shape: [window_size], True means masked out, False means unmasked
        mask = torch.rand(self.window_size) < self.mask_ratio
        sliced_data["mask"] = mask

        return sliced_data


def get_dataloader(data_dir, seq_len=32, batch_size=32, use_subset=False):
    """
    Initializes PretrainingDataset and returns a PyTorch DataLoader.
    """
    from torch.utils.data import DataLoader

    dataset = PretrainingDataset(
        data_dir=data_dir, window_size=seq_len, mask_ratio=0.5, use_subset=use_subset
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
