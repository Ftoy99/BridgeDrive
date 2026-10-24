"""Add zero-init sigma_mlp keys to the released BridgeDrive K=60 ckpt -> HeavyBridge-loadable ckpt.

Output with sigma_mlp = 0 is exactly the Gaussian BridgeDrive model (sigma input is ignored).
Levy lambda needs fine-tuning to be meaningful; this file is only for smoke tests and as warm start.
Usage: python test/make_heavybridge_init_ckpt.py SRC_CKPT OUT_DIR
"""
import json
import os
import shutil
import sys

import torch

SRC, OUT_DIR = sys.argv[1], sys.argv[2]
D_MODEL = 256  # tf_d_model, from time_mlp.1.weight shape (1024, 256)
PREFIX = "planning_decoder.trajectory_head.sigma_mlp."

sd = torch.load(SRC, map_location="cpu", weights_only=True)
assert not any(k.startswith(PREFIX) for k in sd), "source already has sigma_mlp"
new = {
    PREFIX + "0.weight": torch.zeros(D_MODEL, 1),
    PREFIX + "0.bias": torch.zeros(D_MODEL),
    PREFIX + "2.weight": torch.zeros(D_MODEL, D_MODEL),
    PREFIX + "2.bias": torch.zeros(D_MODEL),
}
sd.update(new)
# self-check: every sigma_mlp tensor is zero and only these keys were added
assert all(torch.count_nonzero(v) == 0 for v in new.values())
assert len(sd) == 1618 + len(new), len(sd)

os.makedirs(OUT_DIR, exist_ok=True)
torch.save(sd, os.path.join(OUT_DIR, "model_heavybridge_init_k60.pth"))
shutil.copy(os.path.join(os.path.dirname(SRC), "config.json"), os.path.join(OUT_DIR, "config.json"))
print("wrote", os.path.join(OUT_DIR, "model_heavybridge_init_k60.pth"), "keys", len(sd))
