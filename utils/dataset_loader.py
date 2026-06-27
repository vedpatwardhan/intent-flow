import os
import torch
from torch.utils.data import Dataset, DataLoader


class LatentFlowDataset(Dataset):
    """
    Dataloader designed to ingest pre-tokenized features (DINOv3, CLIP, PointNeXt)
    along with joint states, tactile pressure arrays, and actions.
    """

    def __init__(self, data_dir=None, seq_len=8, mode="train"):
        super().__init__()
        self.seq_len = seq_len
        self.mode = mode

        # Load from disk if dataset is available
        self.use_synthetic = True
        if data_dir and os.path.exists(data_dir):
            self.file_list = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".pt")
            ]
            if len(self.file_list) > 0:
                self.use_synthetic = False

        if self.use_synthetic:
            # Generate mock dataset for validation runs
            self.dataset_size = 200
        else:
            self.dataset_size = len(self.file_list)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.use_synthetic:
            # Generate synthetic pre-tokenized representations
            batch_data = {
                "vision": torch.randn(self.seq_len, 1024),  # DINOv3 visual tokens
                "text": torch.randn(1, 768),  # CLIP instruction tokens
                "pointnext": torch.randn(
                    self.seq_len, 384
                ),  # PointNeXt local point clouds
                "tactile": torch.randn(
                    self.seq_len, 4, 4
                ),  # Spatial fingertip pressure pads
                "proprioception": torch.randn(
                    self.seq_len, 24
                ),  # Joint states [pos, vel]
                "actions": torch.randn(self.seq_len, 12),  # Joint torques
            }
        else:
            # In a full run, load pre-tokenized tensor dictionary from disk
            file_path = self.file_list[idx]
            batch_data = torch.load(file_path)

        return batch_data


def get_dataloader(data_dir=None, seq_len=8, batch_size=32, shuffle=True):
    dataset = LatentFlowDataset(data_dir=data_dir, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
