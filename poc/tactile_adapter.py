import torch
import torch.nn as nn
import torch.optim as optim


class ToyTactileAdapter(nn.Module):
    def __init__(self, in_features=16, out_dim=512):
        super().__init__()
        # A simple MLP to project low-res touch sensor data
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.GELU(), nn.Linear(64, out_dim)
        )

    def forward(self, force_grid):
        # force_grid: [B, in_features] (e.g. 4x4 contact grid)
        return self.net(force_grid)


# InfoNCE Loss for Contrasive Alignment (CASA)
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_a, feat_b):
        # Normalize embeddings
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        # feat_b: [B, out_dim]
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)

        # Calculate similarity logits matrix
        logits = torch.matmul(feat_a, feat_b.T) / self.temp
        labels = torch.arange(feat_a.size(0), device=feat_a.device)

        loss_a = self.cross_entropy(logits, labels)
        loss_b = self.cross_entropy(logits.T, labels)
        return (loss_a + loss_b) / 2.0


def test_tactile_adapter():
    print("=== Task 1.5: Tactile Adapter Calibration PoC ===")
    torch.manual_seed(42)

    B = 32
    tactile_dim = 16  # 4x4 contact pressure grid
    token_dim = 512

    adapter = ToyTactileAdapter(tactile_dim, token_dim)
    info_nce = InfoNCELoss(temperature=0.07)

    # 1. Generate mock MuJoCo touch grids and target vision tokens
    # Contact events are marked by high force values in the grids
    tactile_data = torch.randn(B, tactile_dim) * 0.1  # pre-contact noise
    contact_indices = torch.randint(0, B, (10,))
    # Inject high normal forces (contact impact) at contact indices
    tactile_data[contact_indices] = (
        tactile_data[contact_indices] + torch.randn(10, tactile_dim) * 5.0
    )

    # Target vision tokens (representing contact visually)
    vision_tokens = torch.randn(B, token_dim)
    # Ensure paired indices share semantic alignment
    vision_tokens[contact_indices] = (
        vision_tokens[contact_indices] + torch.randn(10, token_dim) * 2.0
    )

    optimizer = optim.Adam(adapter.parameters(), lr=1e-3)

    # 2. Train the tactile adapter to align with vision tokens during contact
    for epoch in range(50):
        optimizer.zero_grad()
        # Project tactile features
        tactile_tokens = adapter(tactile_data)

        # We only compute InfoNCE alignment loss on active contact frames
        contact_mask = torch.abs(tactile_data).mean(dim=-1) > 0.5

        if contact_mask.sum() > 1:
            loss = info_nce(tactile_tokens[contact_mask], vision_tokens[contact_mask])
            loss.backward()
            optimizer.step()

    # Evaluate similarity on aligned vs. unaligned pairs
    with torch.no_grad():
        final_tactile = adapter(tactile_data)
        final_tactile = final_tactile / final_tactile.norm(dim=-1, keepdim=True)
        final_vision = vision_tokens / vision_tokens.norm(dim=-1, keepdim=True)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(final_tactile, final_vision.T)

        # Mean similarity on matched contact pairs
        matched_sim = torch.mean(torch.diagonal(sim_matrix)[contact_mask]).item()
        # Mean similarity on unmatched pairs
        unmatched_sim = torch.mean(sim_matrix[~torch.eye(B, dtype=torch.bool)]).item()

    print(f"Number of Contact Frames: {contact_mask.sum().item()}")
    print(f"Matched Contact Pairs Similarity: {matched_sim:.4f}")
    print(f"Unmatched Random Pairs Similarity: {unmatched_sim:.4f}")

    # Validation Check
    if matched_sim > unmatched_sim + 0.3:
        print(
            "PoC Result: SUCCESS (Tactile tokens successfully aligned with visual tokens)"
        )
    else:
        print("PoC Result: FAILED (Low alignment similarity)")


if __name__ == "__main__":
    test_tactile_adapter()
