"""
Verification test for encoder/stnet_modules.py. Run from anywhere with:
    <path-to-venv>/python.exe event_rl/scripts/test_stnet_modules.py

Checks, in order:
  1. Shapes match what the docstrings claim, for a real forward pass.
  2. detach_state() lets you actually run multiple PPO-style training steps
     (backward + optimizer step) without the recurrent-graph error.
  3. Reproduces the SAME error WITHOUT detach_state(), proving (1) isn't
     a fluke -- the fix is doing real work, not a no-op.
  4. Reproduces the known first_seq=True + small-crop shape bug, proving
     it's a genuine issue in the original architecture, not a misreading.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "encoder"))

import torch
from stnet_modules import SpatialBranch, TemporalBranch, Fusion

torch.manual_seed(0)
B, H, W = 2, 127, 127
img_pos = [torch.randn(B, 3, H, W) for _ in range(5)]
img_neg = [torch.randn(B, 3, H, W) for _ in range(5)]

print("=" * 70)
print("1. SHAPE CHECK")
print("=" * 70)

spatial, temporal, fusion = SpatialBranch(), TemporalBranch(), Fusion()
spatial.eval(); temporal.eval(); fusion.eval()

with torch.no_grad():
    output_sig, output_lowres = spatial(img_pos, img_neg)
    tem_fea, spa_fea = temporal(img_pos, img_neg, output_lowres, first_seq=False)
    fused = fusion(tem_fea, spa_fea)

print(f"SpatialBranch  : output_sig={tuple(output_sig.shape)}  output_lowres={tuple(output_lowres.shape)}")
print(f"TemporalBranch : tem_fea={tuple(tem_fea.shape)}  spa_fea={tuple(spa_fea.shape)}")
print(f"  hidden state : {len(temporal.state)} tensors, e.g. {tuple(temporal.state[0].shape)}")
print(f"Fusion         : fused={tuple(fused.shape)}")

print()
print("=" * 70)
print("2. TRAINING STEPS WITH detach_state() -- should all succeed")
print("=" * 70)

spatial.train(); temporal.train(); fusion.train()
temporal.reset()
opt = torch.optim.SGD(
    list(spatial.parameters()) + list(temporal.parameters()) + list(fusion.parameters()), lr=1e-3
)
for step in range(3):
    _, lowres = spatial(img_pos, img_neg)
    tem_fea, spa_fea = temporal(img_pos, img_neg, lowres, first_seq=False)
    loss = fusion(tem_fea, spa_fea).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    temporal.detach_state()
    print(f"  step {step}: loss={loss.item():.6f}  OK")
print("PASSED: 3 steps completed with detach_state() between them.")

print()
print("=" * 70)
print("3. SAME LOOP WITHOUT detach_state() -- should FAIL (proves the fix matters)")
print("=" * 70)

temporal.reset()
_, lowres = spatial(img_pos, img_neg)
tem_fea, spa_fea = temporal(img_pos, img_neg, lowres, first_seq=False)
fusion(tem_fea, spa_fea).mean().backward()
try:
    tem_fea2, spa_fea2 = temporal(img_pos, img_neg, lowres, first_seq=False)
    fusion(tem_fea2, spa_fea2).mean().backward()
    print("UNEXPECTED: second backward without detach_state() did not raise!")
except RuntimeError as e:
    print(f"EXPECTED FAILURE (this is correct): {str(e)[:90]}")

print()
print("=" * 70)
print("4. KNOWN ISSUE: first_seq=True + 127px crop -> conv33_11 shape bug")
print("=" * 70)

temporal2 = TemporalBranch()
try:
    with torch.no_grad():
        temporal2(img_pos, img_neg, output_lowres, first_seq=True)
    print("UNEXPECTED: this did not raise!")
except RuntimeError as e:
    print(f"EXPECTED (documented in TemporalBranch's docstring): {str(e)[:100]}")

print()
print("ALL CHECKS COMPLETE.")
