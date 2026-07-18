import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).flatten())

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x.view(bs, self.k, self.k)


class PointNetClassification(nn.Module):
    def __init__(self, num_classes=40, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.feature_transform = TNet(k=64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        bs = x.size(0)
        trans_3x3 = self.input_transform(x)
        x = torch.bmm(trans_3x3, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        trans_64x64 = self.feature_transform(x)
        x = torch.bmm(trans_64x64, x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x, trans_3x3, trans_64x64


class PointNetEncoder(nn.Module):
    def __init__(self, out_channels=384, device="cuda"):
        super().__init__()
        self.pointnet = PointNetClassification(num_classes=40)

        # Download and load pretrained model weights from Hugging Face Hub
        try:
            print(
                "[PointNet Loader] Downloading pretrained weights from DavidHanSZ/pointnet-modelnet40..."
            )
            model_path = hf_hub_download(
                repo_id="DavidHanSZ/pointnet-modelnet40", filename="pytorch_model.bin"
            )
            state_dict = torch.load(model_path, map_location=device)
            self.pointnet.load_state_dict(state_dict)
            print("[PointNet Loader] Pretrained weights loaded successfully.")
        except Exception as e:
            print(
                f"[PointNet Loader] Warning: Failed to load pretrained weights ({e}). Running on random initialization."
            )

        # Freeze pretrained PointNet backbone parameters
        for param in self.pointnet.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(1024, out_channels)

    def forward(self, x):
        # Input shape: [B, 3, N]
        # Replicate pointnet backbone forward pass up to global max pooling
        trans_3x3 = self.pointnet.input_transform(x)
        x = torch.bmm(trans_3x3, x)
        x = F.relu(self.pointnet.bn1(self.pointnet.conv1(x)))
        x = F.relu(self.pointnet.bn2(self.pointnet.conv2(x)))
        trans_64x64 = self.pointnet.feature_transform(x)
        x = torch.bmm(trans_64x64, x)
        x = F.relu(self.pointnet.bn3(self.pointnet.conv3(x)))
        x = F.relu(self.pointnet.bn4(self.pointnet.conv4(x)))
        x = F.relu(self.pointnet.bn5(self.pointnet.conv5(x)))

        # Global max pooled features (B, 1024)
        global_feat = torch.max(x, dim=2, keepdim=False)[0]

        # Project to target dimension (B, 384) and unsqueeze for point-pooling interface compatibility
        out = self.proj(global_feat).unsqueeze(-1)
        return out
