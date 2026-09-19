import os, sys, struct, glob
import torch
from PIL import Image
from safetensors import safe_open
from tokenizers import Tokenizer
import numpy as np

print("=================================================================================")
print(" 🌌 AETHER-VL :: EXTRACTOR DE FOTONES Y PROMPT DIRECTO (005.jpg)")
print("=================================================================================")

img_path = "/Users/crotalo/Downloads/005.jpg"
if not os.path.exists(img_path):
    print(f"❌ Error: {img_path} no existe.")
    sys.exit(1)

model_dir = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

print("• Cargando tensores de Vision Tower en FP32...")
weights = {}
for s in shards:
    with safe_open(s, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "vision_tower" in k and any(x in k for x in ["patch_embed", "merger"]):
                weights[k] = f.get_tensor(k).float()

print(f"✓ {len(weights)} tensores cargados en FP32.")

img = Image.open(img_path).convert("RGB")
W_orig, H_orig = img.size
print(f"✓ Imagen original: {W_orig}x{H_orig}")

# Geometría nativa Qwen: parche 16, merge 2 (factor 32)
patch_size = 16
merge_size = 2
factor = 32

H_new = max(factor, (H_orig // factor) * factor)
W_new = max(factor, (W_orig // factor) * factor)

max_dim = 768
if max(H_new, W_new) > max_dim:
    scale = max_dim / max(H_new, W_new)
    H_new = int((H_new * scale) // factor) * factor
    W_new = int((W_new * scale) // factor) * factor

img_resized = img.resize((W_new, H_new), Image.Resampling.BICUBIC)
print(f"✓ Imagen calibrada a: {W_new}x{H_new} (Grid: {H_new//patch_size}x{W_new//patch_size})")

img_arr = np.array(img_resized, dtype=np.float32) / 127.5 - 1.0

grid_h = H_new // patch_size
grid_w = W_new // patch_size

patches = []
for gh in range(grid_h):
    for gw in range(grid_w):
        y0, y1 = gh * patch_size, (gh + 1) * patch_size
        x0, x1 = gw * patch_size, (gw + 1) * patch_size
        p = img_arr[y0:y1, x0:x1, :]
        p_temp = np.stack([p, p], axis=0) # temporal = 2
        patches.append(p_temp.flatten())

pixel_values = torch.tensor(np.array(patches), dtype=torch.float32) # [1824, 1536]

# Proyección patch_embed
proj_w = weights["vision_tower.patch_embed.proj.weight"].reshape(1152, 1536)
proj_b = weights["vision_tower.patch_embed.proj.bias"]

p_1152 = torch.matmul(pixel_values, proj_w.T) + proj_b

# Fusión espacial 2x2
h_merged = grid_h // 2
w_merged = grid_w // 2
n_merged = h_merged * w_merged

p_grid = p_1152.reshape(grid_h, grid_w, 1152)
p_merged = p_grid.reshape(h_merged, 2, w_merged, 2, 1152).permute(0, 2, 1, 3, 4).reshape(n_merged, 4608)

# Merger FC1 -> GELU -> FC2 (D=5120)
fc1_w = weights["vision_tower.merger.linear_fc1.weight"]
fc1_b = weights["vision_tower.merger.linear_fc1.bias"]
fc2_w = weights["vision_tower.merger.linear_fc2.weight"]
fc2_b = weights["vision_tower.merger.linear_fc2.bias"]

h_fc1 = torch.nn.functional.gelu(torch.matmul(p_merged, fc1_w.T) + fc1_b)
visual_tokens = torch.matmul(h_fc1, fc2_w.T) + fc2_b

visual_tokens = visual_tokens.numpy()
print(f"✓ Tokens multimodales proyectados a D=5120: {visual_tokens.shape}")

visual_tokens.tofile("visual_embeddings.bin")
print(f"✓ visual_embeddings.bin generado: {n_merged} parches fácticos ({os.path.getsize('visual_embeddings.bin')} bytes).")

# ─── 2. COMPILAR PROMPT CON PLANTILLA DIRECTA (SIN THINKING LOOP) ───────────
tok_path = os.path.join(model_dir, "tokenizer.json")
tokenizer = Tokenizer.from_file(tok_path)

texto_pre = "<|im_start|>user\nPicture 1: "
texto_post = "\nDescribe detalladamente los elementos, colores, objetos y contexto que se observan en la imagen.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"

tokens_pre = tokenizer.encode(texto_pre).ids
tokens_post = tokenizer.encode(texto_post).ids

tokens_pre.append(248055) # <|vision_start|>
tokens_vision = [248056] * n_merged
tokens_vision.append(248057) # <|vision_end|>

secuencia_final = tokens_pre + tokens_vision + tokens_post

with open("prompt_input.bin", "wb") as f:
    for t in secuencia_final:
        f.write(struct.pack("I", t))

print(f"✓ prompt_input.bin compilado: {len(secuencia_final)} tokens ({n_merged} parches + prefijo cerrado <think>).")
print("=================================================================================")
print(" 🏆 TODO LISTO PARA LA INFERENCIA CON 005.jpg.")
print("=================================================================================")
