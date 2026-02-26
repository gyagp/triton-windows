"""Benchmark forward pass timing with full-size inputs."""
import sys, os, time
sys.path.insert(0, os.path.join("python"))
sys.path.insert(0, os.path.join("python", "examples", "webgpu"))
sys.path.insert(0, os.path.join("python", "examples", "webgpu", "flux-klein"))
os.environ.setdefault("TRITON_WEBGPU_DAWN_PATH",
                      os.path.join("python", "examples", "webgpu", "libs", "dawn.dll"))
import numpy as np
from model import FluxKleinWebGPU

weights_path = os.path.join("python", "examples", "webgpu", "flux-klein",
                            "weights", "transformer_fp16.npz")
weights = dict(np.load(weights_path, allow_pickle=True))
model = FluxKleinWebGPU(weights, fp16_act=False)

T_img, T_txt = 1024, 512
latents = np.random.randn(T_img, 64).astype(np.float32) * 0.01
enc = np.random.randn(T_txt, 3072).astype(np.float32) * 0.01
img_ids = np.zeros((T_img, 4), dtype=np.float32)
txt_ids = np.zeros((T_txt, 4), dtype=np.float32)

# warmup
_ = model.forward(latents, enc, 0.5, img_ids, txt_ids)

# timed runs
times = []
for i in range(5):
    t0 = time.perf_counter()
    _ = model.forward(latents, enc, 0.5, img_ids, txt_ids)
    times.append(time.perf_counter() - t0)
    print(f"  Run {i+1}: {times[-1]*1000:.0f}ms")

print(f"\nBest: {min(times)*1000:.0f}ms, Mean: {sum(times)/len(times)*1000:.0f}ms")
