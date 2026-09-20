import os, sys
import mlx.core as mx
try:
    from . import aether_native_c
except ImportError:
    import aether_native_c
from .settling import run_deep_thought_settling

class AetherCoupledLayer:
    """
    Envoltura de capa que delega la evolución geodésica directamente al módulo C++ nativo.
    """
    def __init__(self, original_layer, layer_idx, num_layers=64, theta_steer=0.35):
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.dt = 1.0 / float(num_layers)
        self.theta_step = theta_steer / float(num_layers)
        self.state_ref = None 
        self.active = False
        self.call_count = 0

    def __getattr__(self, name):
        return getattr(self.original_layer, name)

    def __call__(self, x, **kwargs):
        h = self.original_layer(x, **kwargs)
        if not self.active or self.state_ref is None:
            return h

        u_target = self.state_ref.get("L_star")
        if u_target is None:
            return h

        self.call_count += 1
        # Invocación directa a la extensión C++ nativa
        if h.shape[1] == 1:
            h_mod = aether_native_c.dispatch_riemannian_step(h, u_target, self.theta_step)
            # En la capa terminal (63), registrar el vector de velocidad tangencial v_drag
            if self.layer_idx == (self.num_layers - 1):
                self.state_ref["v_drag"] = (h_mod[0, 0, :] - h[0, 0, :]) / self.dt
            return h_mod
        else:
            h_last = aether_native_c.dispatch_riemannian_step(h[0, -1, :], u_target, self.theta_step)
            if self.layer_idx == (self.num_layers - 1):
                self.state_ref["v_drag"] = (h_last - h[0, -1, :]) / self.dt
            return mx.concatenate([h[:, :-1, :], h_last[None, None, :]], axis=1)

class AetherCollapseHead:
    """
    Capa de Colapso: Ejecuta el Choque Cinético (gamma), la Proyección (W_head)
    y la Condensación de Vapor (nu, kappa) íntegramente en C++ nativo.
    CERO callbacks a Python: los tensores de cuantización se extraen una vez al inicio.
    """
    def __init__(self, original_lm_head, nu=0.12, gamma=0.35, kappa=0.15):
        self.original_lm_head = original_lm_head
        self.nu = nu
        self.gamma = gamma
        self.kappa = kappa
        self.state_ref = None
        self.active = False
        self.call_count = 0
        self.last_raw_logits = None
        self.last_cond_logits = None

        # Extraer buffers de cuantización UNA SOLA VEZ (cero overhead por token)
        self.head_w = original_lm_head.weight
        self.head_scales = original_lm_head.scales
        self.head_biases = original_lm_head.biases
        self.head_group_size = getattr(original_lm_head, 'group_size', 64)
        self.head_bits = getattr(original_lm_head, 'bits', 4)

    def __getattr__(self, name):
        return getattr(self.original_lm_head, name)

    def __call__(self, h):
        if not self.active or self.state_ref is None:
            return self.original_lm_head(h)

        v_drag = self.state_ref.get("v_drag")
        z_L_star = self.state_ref.get("z_L_star")

        if v_drag is None or z_L_star is None:
            return self.original_lm_head(h)

        self.call_count += 1
        # Invocación directa al pipeline completo en C++ nativo
        # Pasa tensores crudos de cuantización — CERO Python en el colapso
        z_condensed = aether_native_c.dispatch_full_collapse(
            h, v_drag, z_L_star,
            self.head_w, self.head_scales, self.head_biases,
            self.head_group_size, self.head_bits,
            self.nu, self.gamma, self.kappa
        )
        self.last_cond_logits = z_condensed
        return z_condensed

class AetherEngine:
    """
    Orquestador soberano en silicio.
    """
    def __init__(self, model, processor, nu=0.12, gamma=0.35, kappa=0.15, theta_steer=0.35):
        self.model = model
        self.processor = processor
        self.nu = nu
        self.gamma = gamma
        self.kappa = kappa
        self.theta_steer = theta_steer
        self.num_layers = len(self.model.language_model.model.layers)
        self.hooked_layers = []
        self.hooked_head = None
        self.state = {}
        self._install_circuit()

    def _install_circuit(self):
        self.hooked_layers = []
        for l in range(self.num_layers):
            orig = self.model.language_model.model.layers[l]
            hook = AetherCoupledLayer(orig, layer_idx=l, num_layers=self.num_layers, theta_steer=self.theta_steer)
            hook.state_ref = self.state
            self.model.language_model.model.layers[l] = hook
            self.hooked_layers.append(hook)

        orig_head = self.model.language_model.lm_head
        self.hooked_head = AetherCollapseHead(orig_head, nu=self.nu, gamma=self.gamma, kappa=self.kappa)
        self.hooked_head.state_ref = self.state
        self.model.language_model.lm_head = self.hooked_head

    def set_active(self, active: bool):
        for hook in self.hooked_layers:
            hook.active = active
        if self.hooked_head:
            self.hooked_head.active = active

    def reset_counters(self):
        for hook in self.hooked_layers:
            hook.call_count = 0
        if self.hooked_head:
            self.hooked_head.call_count = 0
            self.hooked_head.last_raw_logits = None
            self.hooked_head.last_cond_logits = None

    def prepare_multimodal_thought(self, visual_patches, text_prompt):
        # 1. Atractor visual fáctico
        u_vis = mx.mean(visual_patches, axis=0)
        u_vis = u_vis / mx.sqrt(mx.sum(u_vis * u_vis) + 1e-12)

        # 2. Extracción de intención lingüística real
        input_ids = self.processor.tokenizer.encode(text_prompt)
        input_tensor = mx.array(input_ids)[None, :]
        text_embeds = self.model.language_model.model.embed_tokens(input_tensor)[0]
        u_txt = mx.mean(text_embeds, axis=0)
        u_txt = (u_txt / mx.sqrt(mx.sum(u_txt * u_txt) + 1e-12)).astype(mx.float32)

        # 3. Pensamiento Profundo tau* = 32
        L_star, telemetria = run_deep_thought_settling(u_vis, u_txt, tau_steps=32)

        # 4. Pre-cálculo del perfil de nucleación de L* sobre logits en C++
        z_L_star = self.hooked_head.original_lm_head(L_star)
        mx.eval(z_L_star)

        self.state["u_vis"] = u_vis
        self.state["u_txt"] = u_txt
        self.state["L_star"] = L_star
        self.state["z_L_star"] = z_L_star
        self.state["v_drag"] = None

        self.set_active(True)
        return telemetria
