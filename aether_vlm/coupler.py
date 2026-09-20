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
    Capa de Colapso con Honda Gravitacional de Penrose y Métrica Geodésica Intrínseca en S^{D-1}.
    1. Métrica Geodésica: Delta_G = (kappa/2) * arccos^2(cos_theta) en lugar de cuerda plana.
    2. Honda de Penrose: Reflejo elástico de velocidad de caída e impulso ortogonal sobre W_t.
    3. Deflación Gram-Schmidt: El atractor L* se despoja de los conceptos ya radiados.
    Permite operar en resonancia plena (kappa=2.00, theta=1.40) SIN apagar la gravedad.
    """
    def __init__(self, original_lm_head, nu=0.12, gamma=0.35, kappa=0.15, tau_relax=None, slingshot=True):
        self.original_lm_head = original_lm_head
        self.nu = nu
        self.gamma = gamma
        self.kappa = kappa
        self.tau_relax = tau_relax
        self.slingshot = slingshot
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

        # Precalcular matriz des-cuantizada W_head y normas euclídeas de fila
        self.deq_W = mx.dequantize(
            self.head_w, self.head_scales, self.head_biases,
            group_size=self.head_group_size, bits=self.head_bits
        )
        self.row_norms = mx.sqrt(mx.sum(self.deq_W * self.deq_W, axis=-1) + 1e-12)
        mx.eval(self.deq_W)
        mx.eval(self.row_norms)

    def __getattr__(self, name):
        return getattr(self.original_lm_head, name)

    def recompute_geodesic_delta_G(self, target_vector):
        """Calcula el potencial geodésico cuadrático Delta_G = (kappa/2) * arccos^2(<h, w>/||w||)."""
        z_target = self.original_lm_head(target_vector)
        cos_t = mx.clip(z_target / self.row_norms, -1.0 + 1e-7, 1.0 - 1e-7)
        d_g = mx.arccos(cos_t)
        delta_G = 0.5 * self.kappa * mx.square(d_g)
        mx.eval(delta_G)
        return delta_G

    def __call__(self, h):
        if not self.active or self.state_ref is None:
            return self.original_lm_head(h)

        v_drag = self.state_ref.get("v_drag")
        delta_G = self.state_ref.get("delta_G")

        if delta_G is None:
            return self.original_lm_head(h)

        self.call_count += 1

        # Modulación temporal opcional (si tau_relax está explícitamente fijado y no es inf/None)
        t = self.state_ref.get("gen_step", 0)
        tau = self.state_ref.get("tau_relax", self.tau_relax)
        decay = math.exp(-t / tau) if (tau is not None and tau > 0 and tau < 1e8) else 1.0

        delta_G_eff = delta_G * decay
        nu_eff = self.nu * decay
        gamma_eff = self.gamma * decay

        # Invocación al pipeline nativo C++ sobre Metal GPU
        z_condensed = aether_native_c.dispatch_full_collapse(
            h,
            v_drag if v_drag is not None else mx.zeros_like(h[0, 0, :]),
            delta_G_eff,
            self.head_w, self.head_scales, self.head_biases,
            self.head_group_size, self.head_bits,
            nu_eff, gamma_eff
        )
        self.last_cond_logits = z_condensed

        # Dinámica de Honda Gravitacional de Penrose y Deflación de Atractor
        if h.shape[1] == 1:
            self.state_ref["gen_step"] = t + 1

            if self.slingshot:
                # 1. Token colapsado en el periapsis
                token_id = int(mx.argmax(z_condensed[0, -1, :]))
                w_t = self.deq_W[token_id]
                norm_wt = self.row_norms[token_id]
                w_t_unit = w_t / (norm_wt + 1e-12)

                # 2. Deflación Gram-Schmidt de la dimensión semántica ya emitida
                L_curr = self.state_ref.get("L_star")
                if L_curr is not None:
                    proj_L = mx.sum(L_curr * w_t_unit)
                    L_next = L_curr - (proj_L * 0.95) * w_t_unit
                    norm_L = mx.sqrt(mx.sum(L_next * L_next) + 1e-12)
                    L_next = L_next / norm_L
                    self.state_ref["L_star"] = L_next

                    # 3. Honda de Penrose: Reflejo elástico de velocidad de caída
                    if v_drag is not None:
                        v_rad = mx.sum(v_drag * w_t_unit) * w_t_unit
                        v_perp = v_drag - v_rad
                        # v_slingshot: Inversión elástica (-1.5*v_rad) e impulso transversal hacia el nuevo atractor
                        v_slingshot = v_perp - 1.5 * v_rad + (self.gamma * 0.1) * L_next
                        self.state_ref["v_drag"] = v_slingshot

                    # 4. Actualización del potencial geodésico d_g cada 2 tokens (cero overhead)
                    if self.call_count % 2 == 0:
                        self.state_ref["delta_G"] = self.recompute_geodesic_delta_G(L_next)

        return z_condensed

AETHER_MODEL_PROFILES = {
    # Perfil 1: Modelos Edge (0.8B) -> Capacidad delicada, acoplamiento suave
    "edge_compact": {
        "theta_steer": 0.25,
        "kappa": 0.15,
        "gamma": 0.25,
        "nu": 0.08,
        "active_layers_ratio": 0.50,
        "slingshot": True,
    },
    # Perfil 2: Modelos Compactos de atención densa con pesos atados (Qwen3.5-2B)
    "compact_tied": {
        "theta_steer": 1.40,
        "kappa": 2.00,
        "gamma": 0.95,
        "nu": 0.08,
        "active_layers_ratio": 0.50,  # Capas 12..24
        "slingshot": True,
    },
    # Perfil 3: Modelo frontier denso de 64 capas, D=5120 y lm_head dedicado (Qwen3.8-27B)
    "frontier_dense": {
        "theta_steer": 2.20,         # Compensación por dispersión en 64 capas
        "kappa": 1.20,               # Ajuste por concentración hiper-esférica en D=5120
        "gamma": 0.85,
        "nu": 0.06,
        "active_layers_ratio": 0.60, # Capas 26..64 (respeta capas lineales recurrentes tempranas)
        "slingshot": True,
    }
}

class AetherEngine:
    """
    Orquestador soberano en silicio con Honda Gravitacional de Penrose,
    proyección covariante geodésica arccos y autoconfiguración por perfil arquitectónico.
    """
    def __init__(self, model, processor, nu=None, gamma=None, kappa=None, theta_steer=None, tau_relax=None, slingshot=None):
        self.model = model
        self.processor = processor
        self.num_layers = len(self.model.language_model.model.layers)
        
        # Detección de dimensión latente D (desempaquetando 4-bit uint32 si aplica)
        if hasattr(self.model.language_model.model, "embed_tokens"):
            w = getattr(self.model.language_model.model.embed_tokens, "weight", None)
            bits = getattr(self.model.language_model.model.embed_tokens, "bits", 4)
            pack_factor = (32 // bits) if (w is not None and w.dtype == mx.uint32) else 1
            self.hidden_dim = (w.shape[-1] * pack_factor) if w is not None else 2048
        else:
            self.hidden_dim = 2048

        # Autoselección de perfil según topología de capas y dimensión latente
        if self.num_layers >= 48 or self.hidden_dim >= 4096:
            self.profile_name = "frontier_dense"
        elif self.hidden_dim <= 1024:
            self.profile_name = "edge_compact"
        else:
            self.profile_name = "compact_tied"

        profile = AETHER_MODEL_PROFILES[self.profile_name]

        # Asignar parámetros con override explícito si fue provisto
        self.nu = nu if nu is not None else profile["nu"]
        self.gamma = gamma if gamma is not None else profile["gamma"]
        self.kappa = kappa if kappa is not None else profile["kappa"]
        self.theta_steer = theta_steer if theta_steer is not None else profile["theta_steer"]
        self.slingshot = slingshot if slingshot is not None else profile["slingshot"]
        self.active_layers_ratio = profile.get("active_layers_ratio", 0.50)
        self.tau_relax = tau_relax

        self.hooked_layers = []
        self.hooked_head = None
        self.state = {
            "gen_step": 0,
            "tau_relax": tau_relax,
            "slingshot": self.slingshot,
            "profile": self.profile_name
        }
        self._install_circuit()

    def _install_circuit(self):
        self.hooked_layers = []
        # Inicio de capas activas según perfil
        start_active = int(self.num_layers * (1.0 - self.active_layers_ratio))
        num_active = max(1, self.num_layers - start_active)

        for l in range(self.num_layers):
            orig = self.model.language_model.model.layers[l]
            hook = AetherCoupledLayer(
                orig, layer_idx=l, num_layers=self.num_layers,
                theta_steer=self.theta_steer, tau_relax=self.tau_relax
            )
            if l < start_active:
                hook.theta_step = 0.0
            else:
                progress = (l - start_active + 1) / float(num_active)
                hook.theta_step = (self.theta_steer / float(num_active)) * progress
            hook.state_ref = self.state
            self.model.language_model.model.layers[l] = hook
            self.hooked_layers.append(hook)

        if hasattr(self.model.language_model, "lm_head") and self.model.language_model.lm_head is not None:
            orig_head = self.model.language_model.lm_head
        else:
            orig_head = TiedLinearHead(self.model.language_model.model.embed_tokens)
            if hasattr(self.model.language_model, "args") and hasattr(self.model.language_model.args, "tie_word_embeddings"):
                self.model.language_model.args.tie_word_embeddings = False

        self.hooked_head = AetherCollapseHead(
            orig_head, nu=self.nu, gamma=self.gamma, kappa=self.kappa, tau_relax=self.tau_relax, slingshot=self.slingshot
        )
        self.hooked_head.state_ref = self.state
        self.model.language_model.lm_head = self.hooked_head

    def update_parameters(self, theta_steer=None, gamma=None, kappa=None, nu=None, tau_relax=None, slingshot=None):
        """Actualiza parámetros dinámicamente recalculando la barrera de Gibbs geodésica."""
        if slingshot is not None:
            self.slingshot = slingshot
            self.hooked_head.slingshot = slingshot
            self.state["slingshot"] = slingshot
        if tau_relax is not None:
            self.tau_relax = tau_relax
            self.state["tau_relax"] = tau_relax
            self.hooked_head.tau_relax = tau_relax
            for hook in self.hooked_layers:
                hook.tau_relax = tau_relax
        if theta_steer is not None:
            self.theta_steer = theta_steer
            start_active = int(self.num_layers * (1.0 - self.active_layers_ratio))
            num_active = max(1, self.num_layers - start_active)
            for l, hook in enumerate(self.hooked_layers):
                if l < start_active:
                    hook.theta_step = 0.0
                else:
                    progress = (l - start_active + 1) / float(num_active)
                    hook.theta_step = (self.theta_steer / float(num_active)) * progress
        if gamma is not None:
            self.gamma = gamma
            self.hooked_head.gamma = gamma
        if nu is not None:
            self.nu = nu
            self.hooked_head.nu = nu
        if kappa is not None:
            self.kappa = kappa
            self.hooked_head.kappa = kappa
            if "L_star" in self.state and self.hooked_head.row_norms is not None:
                self.state["delta_G"] = self.hooked_head.recompute_geodesic_delta_G(self.state["L_star"])

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

    def prepare_multimodal_thought(self, visual_patches, text_prompt, tau_sharp=4.0):
        # 1. Extracción de intención lingüística como lente de polarización
        tok = getattr(self.processor, "tokenizer", self.processor)
        input_ids = tok.encode(text_prompt)
        input_tensor = mx.array(input_ids)[None, :]
        text_embeds = self.model.language_model.model.embed_tokens(input_tensor)[0]
        u_txt = mx.mean(text_embeds, axis=0)
        u_txt = (u_txt / mx.sqrt(mx.sum(u_txt * u_txt) + 1e-12)).astype(mx.float32)

        # 2. Refracción óptica y agudización de foco espacial tau_sharp = 4.0
        vis_norms = mx.sqrt(mx.sum(visual_patches * visual_patches, axis=-1, keepdims=True) + 1e-12)
        vis_unit = visual_patches / vis_norms
        cos_sim = mx.sum(vis_unit * u_txt, axis=-1)
        attn_weights = mx.softmax(tau_sharp * cos_sim, axis=-1)
        S_focal = mx.sum(visual_patches * attn_weights[:, None], axis=0)
        u_vis = S_focal / mx.sqrt(mx.sum(S_focal * S_focal) + 1e-12)

        # 3. Asentamiento Riemanniano disipativo del atractor perceptual puro S*
        S_star, telemetria = run_deep_thought_settling(u_vis, u_vis, tau_steps=32)

        # 4. Potencial Geodésico Intrínseco en S^{D-1} sobre H+ alimentado con S*
        delta_G = self.hooked_head.recompute_geodesic_delta_G(S_star)

        self.state["gen_step"] = 0
        self.state["tau_relax"] = self.tau_relax
        self.state["slingshot"] = self.slingshot
        self.state["u_vis"] = u_vis
        self.state["u_txt"] = u_txt
        self.state["L_star"] = S_star
        self.state["delta_G"] = delta_G
        self.state["v_drag"] = None

        self.set_active(True)
        return telemetria
