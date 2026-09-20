import os, sys, math
import mlx.core as mx
try:
    from . import aether_native_c
except ImportError:
    import aether_native_c
from .settling import run_deep_thought_settling

class AetherCoupledLayer:
    """
    Envoltura de capa que delega la evolución geodésica directamente al módulo C++ nativo.
    Aplica radiación de Hawking del atractor: modulación geodésica decayendo con escala tau_relax.
    """
    def __init__(self, original_layer, layer_idx, num_layers=64, theta_steer=0.35, tau_relax=45.0):
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.dt = 1.0 / float(num_layers)
        self.theta_step = theta_steer / float(num_layers)
        self.tau_relax = tau_relax
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

        # Radiación de Hawking: descarga de entalpía y contracción del radio de Schwarzschild r_s(t) -> 0
        t = self.state_ref.get("gen_step", 0)
        tau = self.state_ref.get("tau_relax", self.tau_relax)
        decay = math.exp(-t / tau) if (tau is not None and tau > 0) else 1.0
        effective_theta_step = self.theta_step * decay

        # Invocación directa a la extensión C++ nativa
        if h.shape[1] == 1:
            h_mod = aether_native_c.dispatch_riemannian_step(h, u_target, effective_theta_step)
            # En la capa terminal, registrar el vector de desplazamiento tangencial v_drag
            if self.layer_idx == (self.num_layers - 1):
                self.state_ref["v_drag"] = h_mod[0, 0, :] - h[0, 0, :]
            return h_mod
        else:
            h_last = aether_native_c.dispatch_riemannian_step(h[0, -1, :], u_target, effective_theta_step)
            if self.layer_idx == (self.num_layers - 1):
                self.state_ref["v_drag"] = h_last - h[0, -1, :]
            return mx.concatenate([h[:, :-1, :], h_last[None, None, :]], axis=1)

class TiedLinearHead:
    """
    Envoltura para modelos con pesos de proyección atados al embedding (tie_word_embeddings=True)
    como Qwen3.5 0.8B y 2B. Permite acceder a los tensores de cuantización y actuar como lm_head.
    """
    def __init__(self, embed_tokens):
        self.embed_tokens = embed_tokens
        self.weight = embed_tokens.weight
        self.scales = embed_tokens.scales
        self.biases = getattr(embed_tokens, "biases", None)
        self.group_size = getattr(embed_tokens, "group_size", 64)
        self.bits = getattr(embed_tokens, "bits", 4)

    def __call__(self, h):
        return self.embed_tokens.as_linear(h)

class AetherCollapseHead:
    """
    Capa de Colapso: Confinamiento Conforme en H+ y Condensación de Gibbs.
    Resuelve la anisotropía del vocabulario mediante proyección esférica covariante S^{D-1}.
    Elimina la singularidad monótona ('el el el...') y previene horizontes de Schwarzschild
    mediante Radiación de Hawking (descarga entrópica dE/dt < 0 con escala tau_relax).
    """
    def __init__(self, original_lm_head, nu=0.12, gamma=0.35, kappa=0.15, tau_relax=45.0):
        self.original_lm_head = original_lm_head
        self.nu = nu
        self.gamma = gamma
        self.kappa = kappa
        self.tau_relax = tau_relax
        self.state_ref = None
        self.active = False
        self.call_count = 0
        self.last_raw_logits = None
        self.last_cond_logits = None

        # Extraer buffers de cuantización UNA SOLA VEZ (cero overhead por token)
        self.head_w = original_lm_head.weight
        self.head_scales = original_lm_head.scales
        self.head_biases = getattr(original_lm_head, 'biases', None)
        self.head_group_size = getattr(original_lm_head, 'group_size', 64)
        self.head_bits = getattr(original_lm_head, 'bits', 4)

        # Precalcular normas euclídeas de las filas de W_head para invarianza covariante en S^{D-1}
        deq = mx.dequantize(
            self.head_w, self.head_scales, self.head_biases,
            group_size=self.head_group_size, bits=self.head_bits
        )
        self.row_norms = mx.sqrt(mx.sum(deq * deq, axis=-1) + 1e-12)
        mx.eval(self.row_norms)

    def __getattr__(self, name):
        return getattr(self.original_lm_head, name)

    def __call__(self, h):
        if not self.active or self.state_ref is None:
            return self.original_lm_head(h)

        v_drag = self.state_ref.get("v_drag")
        delta_G = self.state_ref.get("delta_G")

        # Si kappa fue modificado dinámicamente en el objeto, recalcular delta_G con la nueva escala
        if delta_G is not None and getattr(self, "_last_kappa", None) != self.kappa:
            cos_theta = self.state_ref.get("cos_theta")
            if cos_theta is not None:
                delta_G = 0.5 * self.kappa * mx.square(1.0 - cos_theta)
                self.state_ref["delta_G"] = delta_G
                self._last_kappa = self.kappa

        if delta_G is None:
            return self.original_lm_head(h)

        self.call_count += 1

        # Radiación de Hawking: Descarga de entalpía y evaporación del horizonte r_s(t) -> 0
        t = self.state_ref.get("gen_step", 0)
        tau = self.state_ref.get("tau_relax", self.tau_relax)
        decay = math.exp(-t / tau) if (tau is not None and tau > 0) else 1.0

        delta_G_eff = delta_G * decay
        nu_eff = self.nu * decay
        gamma_eff = self.gamma * decay

        # Invocación directa al pipeline de condensación covariante en C++ nativo (H+)
        # Pasa tensores crudos de cuantización y barrera de Gibbs evaporativa
        z_condensed = aether_native_c.dispatch_full_collapse(
            h,
            v_drag if v_drag is not None else mx.zeros_like(h[0, 0, :]),
            delta_G_eff,
            self.head_w, self.head_scales, self.head_biases,
            self.head_group_size, self.head_bits,
            nu_eff, gamma_eff
        )
        self.last_cond_logits = z_condensed

        # Incrementar el paso de generación propio para tokens autorregresivos (h.shape[1] == 1)
        if h.shape[1] == 1:
            self.state_ref["gen_step"] = t + 1

        return z_condensed

class AetherEngine:
    """
    Orquestador soberano en silicio con proyección covariante en H+
    y radiación de Hawking del horizonte de Schwarzschild.
    """
    def __init__(self, model, processor, nu=0.12, gamma=0.35, kappa=0.15, theta_steer=0.35, tau_relax=45.0):
        self.model = model
        self.processor = processor
        self.nu = nu
        self.gamma = gamma
        self.kappa = kappa
        self.theta_steer = theta_steer
        self.tau_relax = tau_relax
        self.num_layers = len(self.model.language_model.model.layers)
        self.hooked_layers = []
        self.hooked_head = None
        self.state = {"gen_step": 0, "tau_relax": tau_relax}
        self._install_circuit()

    def _install_circuit(self):
        self.hooked_layers = []
        half = self.num_layers // 2
        for l in range(self.num_layers):
            orig = self.model.language_model.model.layers[l]
            hook = AetherCoupledLayer(
                orig, layer_idx=l, num_layers=self.num_layers,
                theta_steer=self.theta_steer, tau_relax=self.tau_relax
            )
            # Modulación de capas: capas tempranas (0..half-1) theta=0 (preserva sintaxis local)
            # Capas tardías (half..num_layers-1): rampa progresiva de síntesis semántica
            if l < half:
                hook.theta_step = 0.0
            else:
                progress = (l - half + 1) / float(self.num_layers - half)
                hook.theta_step = (self.theta_steer / float(self.num_layers - half)) * progress
            hook.state_ref = self.state
            self.model.language_model.model.layers[l] = hook
            self.hooked_layers.append(hook)

        # Detección y adaptación de lm_head (soporta modelos con lm_head y tied-embeddings como Qwen3.5 0.8B/2B)
        if hasattr(self.model.language_model, "lm_head") and self.model.language_model.lm_head is not None:
            orig_head = self.model.language_model.lm_head
        else:
            orig_head = TiedLinearHead(self.model.language_model.model.embed_tokens)
            if hasattr(self.model.language_model, "args") and hasattr(self.model.language_model.args, "tie_word_embeddings"):
                self.model.language_model.args.tie_word_embeddings = False

        self.hooked_head = AetherCollapseHead(
            orig_head, nu=self.nu, gamma=self.gamma, kappa=self.kappa, tau_relax=self.tau_relax
        )
        self.hooked_head.state_ref = self.state
        self.model.language_model.lm_head = self.hooked_head

    def update_parameters(self, theta_steer=None, gamma=None, kappa=None, nu=None, tau_relax=None):
        """Actualiza parámetros dinámicamente recalculando la barrera de Gibbs, modulación geodésica y relajación."""
        if tau_relax is not None:
            self.tau_relax = tau_relax
            self.state["tau_relax"] = tau_relax
            self.hooked_head.tau_relax = tau_relax
            for hook in self.hooked_layers:
                hook.tau_relax = tau_relax
        if theta_steer is not None:
            self.theta_steer = theta_steer
            half = self.num_layers // 2
            for l, hook in enumerate(self.hooked_layers):
                if l < half:
                    hook.theta_step = 0.0
                else:
                    progress = (l - half + 1) / float(self.num_layers - half)
                    hook.theta_step = (self.theta_steer / float(self.num_layers - half)) * progress
        if gamma is not None:
            self.gamma = gamma
            self.hooked_head.gamma = gamma
        if nu is not None:
            self.nu = nu
            self.hooked_head.nu = nu
        if kappa is not None:
            self.kappa = kappa
            self.hooked_head.kappa = kappa
            if "z_L_star" in self.state and self.hooked_head.row_norms is not None:
                cos_theta = self.state["z_L_star"] / self.hooked_head.row_norms
                self.state["delta_G"] = 0.5 * self.kappa * mx.square(1.0 - cos_theta)
                mx.eval(self.state["delta_G"])

    def set_active(self, active: bool):
        for hook in self.hooked_layers:
            hook.active = active
        if self.hooked_head:
            self.hooked_head.active = active

    def reset_counters(self):
        self.state["gen_step"] = 0
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

        # 4. Proyección conforme covariante en S^{D-1} sobre H+
        # Invarianza de norma radial: cos(theta_i) = <w_i, L*> / ||w_i||
        z_L_star = self.hooked_head.original_lm_head(L_star)
        cos_theta = z_L_star / self.hooked_head.row_norms
        
        # Barrera de Gibbs confinada al cono positivo R+: Delta_G = (kappa/2) * (1 - cos(theta))^2
        delta_G = 0.5 * self.kappa * mx.square(1.0 - cos_theta)
        mx.eval(z_L_star)
        mx.eval(delta_G)

        self.state["gen_step"] = 0
        self.state["tau_relax"] = self.tau_relax
        self.state["u_vis"] = u_vis
        self.state["u_txt"] = u_txt
        self.state["L_star"] = L_star
        self.state["z_L_star"] = z_L_star
        self.state["cos_theta"] = cos_theta
        self.state["delta_G"] = delta_G
        self.state["v_drag"] = None

        self.set_active(True)
        return telemetria
