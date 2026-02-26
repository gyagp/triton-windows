"""Compare GPU-resident double blocks vs CPU fallback for correctness."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python", "examples", "webgpu"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python", "examples", "webgpu", "flux-klein"))
os.environ.setdefault("TRITON_WEBGPU_DAWN_PATH",
                      os.path.join(os.path.dirname(__file__), "python", "examples", "webgpu", "libs", "dawn.dll"))

import numpy as np
from model import FluxKleinWebGPU, HIDDEN_DIM, IN_CHANNELS, JOINT_ATTN_DIM, NUM_HEADS, HEAD_DIM
from model import NUM_DOUBLE_BLOCKS, NUM_SINGLE_BLOCKS, FF_DIM, TIMESTEP_CHANNELS
from model import compute_rope_freqs, prepare_latent_ids, prepare_text_ids, AXES_DIMS_ROPE, ROPE_THETA

# Create small random model
rng = np.random.RandomState(42)
s = 0.02
W = {}
D = HIDDEN_DIM

W["time_guidance_embed.timestep_embedder.linear_1.weight"] = rng.randn(D, TIMESTEP_CHANNELS).astype(np.float32) * s
W["time_guidance_embed.timestep_embedder.linear_2.weight"] = rng.randn(D, D).astype(np.float32) * s
W["double_stream_modulation_img.linear.weight"] = rng.randn(D * 6, D).astype(np.float32) * s
W["double_stream_modulation_txt.linear.weight"] = rng.randn(D * 6, D).astype(np.float32) * s
W["single_stream_modulation.linear.weight"] = rng.randn(D * 3, D).astype(np.float32) * s
W["x_embedder.weight"] = rng.randn(D, IN_CHANNELS).astype(np.float32) * s
W["context_embedder.weight"] = rng.randn(D, JOINT_ATTN_DIM).astype(np.float32) * s

for i in range(NUM_DOUBLE_BLOCKS):
    pfx = f"transformer_blocks.{i}"
    for proj in ["to_q", "to_k", "to_v"]:
        W[f"{pfx}.attn.{proj}.weight"] = rng.randn(D, D).astype(np.float32) * s
    W[f"{pfx}.attn.to_out.0.weight"] = rng.randn(D, D).astype(np.float32) * s
    for proj in ["add_q_proj", "add_k_proj", "add_v_proj"]:
        W[f"{pfx}.attn.{proj}.weight"] = rng.randn(D, D).astype(np.float32) * s
    W[f"{pfx}.attn.to_add_out.weight"] = rng.randn(D, D).astype(np.float32) * s
    for norm in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]:
        W[f"{pfx}.attn.{norm}.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
    W[f"{pfx}.ff.linear_in.weight"] = rng.randn(FF_DIM * 2, D).astype(np.float32) * s
    W[f"{pfx}.ff.linear_out.weight"] = rng.randn(D, FF_DIM).astype(np.float32) * s
    W[f"{pfx}.ff_context.linear_in.weight"] = rng.randn(FF_DIM * 2, D).astype(np.float32) * s
    W[f"{pfx}.ff_context.linear_out.weight"] = rng.randn(D, FF_DIM).astype(np.float32) * s

for i in range(NUM_SINGLE_BLOCKS):
    pfx = f"single_transformer_blocks.{i}"
    fused = 3 * D + 2 * FF_DIM
    W[f"{pfx}.attn.to_qkv_mlp_proj.weight"] = rng.randn(fused, D).astype(np.float32) * s
    W[f"{pfx}.attn.norm_q.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
    W[f"{pfx}.attn.norm_k.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
    W[f"{pfx}.attn.to_out.weight"] = rng.randn(D, D + FF_DIM).astype(np.float32) * s

W["norm_out.linear.weight"] = rng.randn(D * 2, D).astype(np.float32) * s
W["proj_out.weight"] = rng.randn(IN_CHANNELS, D).astype(np.float32) * s

print(f"Created {len(W)} tensors")
model = FluxKleinWebGPU(W)
print("Model compiled.\n")

# Inputs
T_img = 16
T_txt = 8
latents = rng.randn(T_img, IN_CHANNELS).astype(np.float32) * 0.1
ctx = rng.randn(T_txt, JOINT_ATTN_DIM).astype(np.float32) * 0.1
img_ids = prepare_latent_ids(4, 4)
txt_ids = prepare_text_ids(T_txt)

# Run GPU-resident path (current forward)
print("=== GPU-resident forward ===")
out_gpu = model.forward(latents, ctx, 0.5, img_ids, txt_ids)
print(f"  shape={out_gpu.shape}, mean={out_gpu.mean():.6f}, std={out_gpu.std():.6f}")
print(f"  range=[{out_gpu.min():.6f}, {out_gpu.max():.6f}]")
print(f"  first 5: {out_gpu[0, :5]}")

# Now run CPU-only path: manually call forward with cpu double blocks
# Patch the model to force CPU double blocks
print("\n=== CPU fallback forward ===")
runner = model.cache.runner

# Manually reconstruct the forward with CPU double blocks
from model import get_timestep_embedding, layer_norm_cpu, EPS
temb = model._compute_temb(0.5)
mod_img = model._compute_modulation(temb, "double_stream_modulation_img.linear.weight", num_param_sets=2)
mod_txt = model._compute_modulation(temb, "double_stream_modulation_txt.linear.weight", num_param_sets=2)
mod_single = model._compute_modulation(temb, "single_stream_modulation.linear.weight", num_param_sets=1)

hidden_states = model._linear(latents, "x_embedder.weight", HIDDEN_DIM)
ctx_np = model._linear(ctx, "context_embedder.weight", HIDDEN_DIM)

image_rope = compute_rope_freqs(img_ids, AXES_DIMS_ROPE, ROPE_THETA)
text_rope = compute_rope_freqs(txt_ids, AXES_DIMS_ROPE, ROPE_THETA)
concat_cos = np.concatenate([text_rope[0], image_rope[0]], axis=0)
concat_sin = np.concatenate([text_rope[1], image_rope[1]], axis=0)

# CPU double blocks
for i in range(NUM_DOUBLE_BLOCKS):
    ctx_np, hidden_states = model._double_block(
        hidden_states, ctx_np, mod_img, mod_txt,
        concat_cos, concat_sin, i, gpu_state=False)

# Concat, single blocks, output
hs = np.concatenate([ctx_np, hidden_states], axis=0)
upload_dt = np.float32
hs_gpu = runner.upload_to_gpu(hs.astype(upload_dt), "tmp_cmp_hs")
hs_gpu.shape = hs.shape
T_total = T_txt + T_img

shift, scale, gate = mod_single[0]
sb_scale_gpu = runner.upload_to_gpu(scale.ravel().astype(np.float32), "cmp_sb_scale")
sb_shift_gpu = runner.upload_to_gpu(shift.ravel().astype(np.float32), "cmp_sb_shift")
sb_gate_gpu = runner.upload_to_gpu(gate.ravel().astype(np.float32), "cmp_sb_gate")

runner.begin_batch()
for i in range(NUM_SINGLE_BLOCKS):
    hs_gpu = model._single_block(hs_gpu, (sb_shift_gpu, sb_scale_gpu, sb_gate_gpu),
                                  concat_cos, concat_sin, i, gpu_state=True)
rb = runner.end_batch(readback_buffers=[hs_gpu])
hidden_states_cpu = rb[id(hs_gpu)].reshape(hs_gpu.shape)
hidden_states_cpu = hidden_states_cpu[T_txt:]
out_cpu = model._output_layer(hidden_states_cpu, temb)

print(f"  shape={out_cpu.shape}, mean={out_cpu.mean():.6f}, std={out_cpu.std():.6f}")
print(f"  range=[{out_cpu.min():.6f}, {out_cpu.max():.6f}]")
print(f"  first 5: {out_cpu[0, :5]}")

# Compare
diff = np.abs(out_gpu - out_cpu)
print(f"\n=== Comparison ===")
print(f"  Max abs diff: {diff.max():.8f}")
print(f"  Mean abs diff: {diff.mean():.8f}")
print(f"  Relative diff: {(diff / (np.abs(out_cpu) + 1e-8)).mean():.6f}")

if diff.max() < 0.01:
    print("  PASS: GPU and CPU paths match closely")
else:
    print("  FAIL: GPU and CPU paths diverge significantly")
    # Find where the biggest diffs are
    worst_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"  Worst diff at {worst_idx}: gpu={out_gpu[worst_idx]:.6f} cpu={out_cpu[worst_idx]:.6f}")
