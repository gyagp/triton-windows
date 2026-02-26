"""Quick FLUX profiling — one forward pass with _profile=True."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "python", "examples", "webgpu"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "python", "examples", "webgpu",
                                "flux-klein"))
os.environ.setdefault("TRITON_WEBGPU_DAWN_PATH",
                      os.path.join(os.path.dirname(__file__),
                                   "python", "examples", "webgpu",
                                   "libs", "dawn.dll"))

import numpy as np

# Load model
from model import FluxKleinWebGPU

weights_path = os.path.join(os.path.dirname(__file__),
                            "python", "examples", "webgpu",
                            "flux-klein", "weights", "transformer_fp16.npz")
print("Loading weights...")
weights = dict(np.load(weights_path, allow_pickle=True))
print(f"Loaded {len(weights)} tensors")

model = FluxKleinWebGPU(weights, fp16_act=False)
print("Model compiled.\n")

# Create dummy inputs matching 512×512
T_img = 1024
T_txt = 512
IN_CHANNELS = 128
JOINT_ATTN_DIM = 7680

latents = np.random.randn(T_img, IN_CHANNELS).astype(np.float32) * 0.01
encoder_hidden_states = np.random.randn(T_txt, JOINT_ATTN_DIM).astype(np.float32) * 0.01
timestep = 0.5
img_ids = np.zeros((T_img, 4), dtype=np.float32)
txt_ids = np.zeros((T_txt, 4), dtype=np.float32)

# Warmup
print("Warmup...")
_ = model.forward(latents, encoder_hidden_states, timestep, img_ids, txt_ids)
print()

# Profiled run
print("Profiled forward pass:")
_ = model.forward(latents, encoder_hidden_states, timestep, img_ids, txt_ids, _profile=True)
