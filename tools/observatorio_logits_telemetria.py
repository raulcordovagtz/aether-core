import os, sys, math, time
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")

print("=================================================================================")
print(" 🔍 PARTE 1: AUTOPSIA DEL PREFILL EN src/main_aether.mm (¿DÓNDE ESTÁ LA IMAGEN?)")
print("=================================================================================\n")

with open("src/main_aether.mm") as f:
    lines = f.readlines()

print("Fragmento del bucle de prefill (Líneas 400 a 425):")
for i in range(399, min(425, len(lines))):
    print(f"  L{i+1}: {lines[i]}", end="")

print("\n---------------------------------------------------------------------------------")
print("🚨 DIAGNÓSTICO EN CÓDIGO:")
print("En L406: [enc setComputePipelineState:psoEmbed];")
print("En L407: [enc setBytes:&tid length:sizeof(uint32_t) atIndex:0];")
print("-> NUNCA se inyecta bufVisualPatches en bufZ cuando tid == 248056 (<|image_pad|>).")
print("-> Las 64 capas de atención procesaron 432 tokens DUMMY de texto repetidos.")
print("---------------------------------------------------------------------------------\n")

print("=================================================================================")
print(" 🔬 PARTE 2: TELEMETRÍA VARIABLE POR VARIABLE SOBRE 248,320 LOGITS")
print("=================================================================================\n")

from core_vlm.utils import load

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
print("• Cargando arquitectura Qwen 27B vía MLX nativo en memoria UMA...")
model, processor = load(model_path)
tokenizer = processor.tokenizer
print("✓ Modelo cargado en GPU.")

D = 5120
R = 32
dt = 0.05
M_coeff = 0.25

# 1. Cargar parches visuales reales
vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)

U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_O_text; V_c[:, 0] = u_O_vis

def apply_J(Grad):
    G_s = Grad[:D]; G_l = Grad[D:]
    return np.concatenate([U_c @ (V_c.T @ G_l), - V_c @ (U_c.T @ G_s)])

def compute_E_and_grad(Phi):
    S = Phi[:D]; L = Phi[D:]
    n_s = np.linalg.norm(S) + 1e-12; n_l = np.linalg.norm(L) + 1e-12
    s_hat = S / n_s; l_hat = L / n_l

    cos_v = np.dot(s_hat, u_O_vis)
    cos_t = np.dot(l_hat, u_O_text)
    cos_vl = np.dot(s_hat, l_hat)

    E = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2
    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_O_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)
    return E, np.concatenate([grad_S, grad_L])

def get_logits_mlx(L_vec):
    l_norm = L_vec / (np.linalg.norm(L_vec) + 1e-12)
    l_mx = mx.array(l_norm, dtype=mx.float32)
    
    # Evaluar a través del LM Head real cuantizado de Qwen 27B
    if hasattr(model.language_model, "lm_head"):
        logits_mx = model.language_model.lm_head(l_mx)
    else:
        logits_mx = model.lm_head(l_mx)
        
    mx.eval(logits_mx)
    logits = np.array(logits_mx)
    shift_z = logits - np.max(logits)
    probs = np.exp(shift_z) / np.sum(np.exp(shift_z))
    return logits, probs

# Estado Inicial en tau = 0
Phi = np.concatenate([u_O_vis, u_O_text])
L_0 = Phi[D:].copy()

print("• Proyectando estado inicial τ = 0 en el LM Head de 27B...")
z_0, p_0 = get_logits_mlx(L_0)
top5_0 = np.argsort(z_0)[-5:][::-1]
h_0 = -np.sum(p_0 * np.log(p_0 + 1e-12))
m_0 = z_0[top5_0[0]] - z_0[top5_0[1]]

print("\n=======================================================================================================================")
print(f"ESTADO INICIAL τ = 0: Entropía H = {h_0:.4f} | Margen m = {m_0:.4f}")
print("Top-5 Palabras en el vocabulario en τ = 0:")
for idx in top5_0:
    word = tokenizer.decode([int(idx)])
    print(f"  • Token {idx:<6} ('{word}'): logit = {z_0[idx]:.4f}, prob = {p_0[idx]*100:.2f}%")
print("=======================================================================================================================\n")

print("• Evolucionando el Biespinor y midiendo impacto causal en los 248,320 logits...")
print(f"{'τ':<4} | {'E(τ)':<8} | {'||∇E||':<8} | {'||Φ̇||':<8} | {'||Δz||_2':<10} | {'||Δz||_∞':<10} | {'ΔMargen':<10} | {'Palabra Ganadora (Top-1)'}")
print("-----------------------------------------------------------------------------------------------------------------------")

for step in range(1, 33):
    E_curr, Grad_curr = compute_E_and_grad(Phi)
    flow = apply_J(Grad_curr) - M_coeff * Grad_curr
    vel = np.linalg.norm(flow)
    grad_norm = np.linalg.norm(Grad_curr)

    Phi += dt * flow

    if step in [1, 4, 8, 16, 24, 32]:
        L_curr = Phi[D:]
        z_curr, p_curr = get_logits_mlx(L_curr)
        
        delta_z_2 = np.linalg.norm(z_curr - z_0)
        delta_z_inf = np.max(np.abs(z_curr - z_0))
        
        top5_curr = np.argsort(z_curr)[-5:][::-1]
        m_curr = z_curr[top5_curr[0]] - z_curr[top5_curr[1]]
        delta_m = m_curr - m_0
        winner_word = tokenizer.decode([int(top5_curr[0])])

        print(f"{step:<4} | {E_curr:<8.4f} | {grad_norm:<8.4f} | {vel:<8.4f} | {delta_z_2:<10.4f} | {delta_z_inf:<10.4f} | {delta_m:<+10.4f} | '{winner_word}' (prob: {p_curr[top5_curr[0]]*100:.1f}%)")

print("=======================================================================================================================")
