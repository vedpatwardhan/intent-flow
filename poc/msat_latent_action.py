import torch
import torch.nn as nn
import torch.optim as optim


# Toy Latent Action Encoder (h_psi): compresses state transitions into 1D latent action
class ToyLatentActionEncoder(nn.Module):
    def __init__(self, state_dim=512, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 128), nn.GELU(), nn.Linear(128, latent_dim)
        )

    def forward(self, st, st_next):
        inputs = torch.cat([st, st_next], dim=-1)
        return self.net(inputs)


# Toy Multi-Stream Action Transformer (MSAT) Cross-Attention block
class ToyMSATBlock(nn.Module):
    def __init__(self, token_dim=512, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads, batch_first=True
        )
        self.query_proj = nn.Parameter(torch.randn(1, 1, token_dim))

    def forward(self, modalities_tokens):
        # modalities_tokens: [B, num_modalities, token_dim]
        # Query visual/semantic tokens using cross attention
        B = modalities_tokens.size(0)
        query = self.query_proj.repeat(B, 1, 1)

        # Cross attention forward
        out_tokens, attn_weights = self.attn(
            query, modalities_tokens, modalities_tokens
        )
        return out_tokens, attn_weights


def test_msat_latent_action():
    print("=== Task 1.3: Latent Action Encoder & MSAT PoC ===")
    torch.manual_seed(42)

    # 1. Verify Latent Action Encoder (h_psi)
    B = 100
    state_dim = 512
    latent_dim = 16

    encoder = ToyLatentActionEncoder(state_dim, latent_dim)

    # Create transition states
    st = torch.randn(B, state_dim)
    st_next = st + torch.randn(B, state_dim) * 0.1  # state delta

    # Extract latent action
    latent_a = encoder(st, st_next)
    print(f"Latent Action output shape: {latent_a.shape}")

    # Check for state leakage: can we reconstruct st from latent_a?
    leakage_decoder = nn.Linear(latent_dim, state_dim)
    opt = optim.Adam(leakage_decoder.parameters(), lr=1e-2)

    # Attempt to train a linear decoder to reconstruct the current state from the latent action
    for _ in range(50):
        opt.zero_grad()
        pred_st = leakage_decoder(latent_a.detach())
        loss = nn.MSELoss()(pred_st, st)
        loss.backward()
        opt.step()

    print(f"State Leakage Reconstruction MSE (Should remain high): {loss.item():.4f}")
    leakage_success = loss.item() > 0.8

    # 2. Verify MSAT Cross-Attention Modality Routing
    msat = ToyMSATBlock(token_dim=512, num_heads=2)

    # Modalities: [CLIP, DINOv3, PointNeXt, Tactile]
    # Case A: Visual Approach (High DINO/PointNeXt features, low Tactile)
    tokens_approach = torch.zeros(1, 4, 512)
    tokens_approach[0, 0] = torch.randn(512) * 0.5  # CLIP Text
    tokens_approach[0, 1] = torch.randn(512) * 1.5  # DINOv3 (Visual Approach)
    tokens_approach[0, 2] = torch.randn(512) * 1.5  # PointNeXt
    tokens_approach[0, 3] = torch.randn(512) * 0.05  # Tactile (No contact)

    _, weights_approach = msat(tokens_approach)
    print(
        f"Approach Attention Weights (CLIP, DINO, PointNeXt, Tactile): {weights_approach[0, 0].detach().numpy()}"
    )

    # Case B: Physical Contact (High Tactile)
    tokens_contact = torch.zeros(1, 4, 512)
    tokens_contact[0, 0] = torch.randn(512) * 0.5  # CLIP Text
    tokens_contact[0, 1] = torch.randn(512) * 0.5  # DINOv3
    tokens_contact[0, 2] = torch.randn(512) * 0.5  # PointNeXt
    tokens_contact[0, 3] = torch.randn(512) * 3.0  # Tactile (Contact impact)

    _, weights_contact = msat(tokens_contact)
    print(
        f"Contact Attention Weights (CLIP, DINO, PointNeXt, Tactile): {weights_contact[0, 0].detach().numpy()}"
    )

    # Validate that attention shifted to tactile during contact
    tactile_attn_approach = weights_approach[0, 0, 3].item()
    tactile_attn_contact = weights_contact[0, 0, 3].item()

    print(
        f"Tactile Attention (Approach vs. Contact): {tactile_attn_approach:.4f} vs. {tactile_attn_contact:.4f}"
    )

    if leakage_success and (tactile_attn_contact > tactile_attn_approach):
        print(
            "PoC Result: SUCCESS (No state leakage detected, attention routing verified)"
        )
    else:
        print("PoC Result: FAILED (Attention routing or anti-leakage failed)")


if __name__ == "__main__":
    test_msat_latent_action()
