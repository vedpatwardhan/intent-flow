import torch
import torch.nn as nn
import torch.optim as optim


# Predictor using standard concatenation (susceptible to action-ignorance collapse)
class ConcatPredictor(nn.Module):
    def __init__(self, state_dim=512, action_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.GELU(), nn.Linear(128, state_dim)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# Predictor using Bilinear Gating (forces action sensitivity)
class GatedPredictor(nn.Module):
    def __init__(self, state_dim=512, action_dim=16):
        super().__init__()
        self.s_proj = nn.Sequential(nn.Linear(state_dim, 128), nn.GELU())
        self.a_proj = nn.Sequential(nn.Linear(action_dim, 128), nn.GELU())
        self.out = nn.Linear(128, state_dim)

    def forward(self, s, a):
        s_feat = self.s_proj(s)
        a_feat = self.a_proj(a)
        # Element-wise gating multiplicative interaction
        gated = s_feat * a_feat
        return self.out(gated)


def evaluate_predictor_collapse():
    print("=== Task 1.4: Predictor Grounding & Sensitivity PoC ===")
    torch.manual_seed(42)

    B = 100
    state_dim = 512
    action_dim = 16

    # Initialize both predictors
    concat_pred = ConcatPredictor(state_dim, action_dim)
    gated_pred = GatedPredictor(state_dim, action_dim)

    # Generate mock inputs
    s_t = torch.randn(B, state_dim)
    a_t = torch.randn(B, action_dim)

    # Target next state: s_{t+1} (depends heavily on action)
    s_next = s_t + a_t[:, :1] * 3.0 + torch.randn(B, state_dim) * 0.1

    # Train both models to fit next state transitions
    optimizer_c = optim.Adam(concat_pred.parameters(), lr=1e-3)
    optimizer_g = optim.Adam(gated_pred.parameters(), lr=1e-3)

    for epoch in range(100):
        # Concatenation training step
        optimizer_c.zero_grad()
        loss_c = nn.MSELoss()(concat_pred(s_t, a_t), s_next)
        loss_c.backward()
        optimizer_c.step()

        # Gated training step
        optimizer_g.zero_grad()
        loss_g = nn.MSELoss()(gated_pred(s_t, a_t), s_next)
        loss_g.backward()
        optimizer_g.step()

    # Evaluate Action Perturbation Drift:
    # We pertub the action input by adding high noise and measure predicted state shift
    a_perturbed = a_t + torch.randn(B, action_dim) * 5.0

    with torch.no_grad():
        c_normal = concat_pred(s_t, a_t)
        c_perturbed = concat_pred(s_t, a_perturbed)
        concat_drift = torch.mean(torch.abs(c_perturbed - c_normal)).item()

        g_normal = gated_pred(s_t, a_t)
        g_perturbed = gated_pred(s_t, a_perturbed)
        gated_drift = torch.mean(torch.abs(g_perturbed - g_normal)).item()

    print(f"Concatenation Predictor Action Drift: {concat_drift:.4f}")
    print(
        f"Bilinearly Gated Predictor Action Drift (Should be high): {gated_drift:.4f}"
    )

    # Success check: Bilinear Gating must show higher action sensitivity
    if gated_drift > concat_drift * 1.5:
        print(
            "PoC Result: SUCCESS (Bilinear Gating prevents copy-paste action-ignorance)"
        )
    else:
        print(
            "PoC Result: FAILED (Gated Predictor failed to maintain action sensitivity)"
        )


if __name__ == "__main__":
    evaluate_predictor_collapse()
