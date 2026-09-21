import mlx.core as mx
from .kernels import dispatch_riemannian_step, dispatch_alu_cell
from .settling import run_deep_thought_settling

class AetherCoupledLayer:
    """
    Envoltura continua que aplica el paso geodésico infinitesimal en la capa l.
    dt = 1.0 / num_layers
    """
    def __init__(self, original_layer, layer_idx, num_layers=64):
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.dt = 1.0 / float(num_layers)
        self.u_target = None
        self.theta_step = 0.35 / float(num_layers) # Rotación acumulativa suave
        self.active = False

    def __getattr__(self, name):
        return getattr(self.original_layer, name)

    def __call__(self, x, **kwargs):
        h = self.original_layer(x, **kwargs)
        if not self.active or self.u_target is None:
            return h

        if h.shape[1] == 1:
            # Decode: Paso geodésico infinitesimal continuo en GPU
            h_mod = dispatch_riemannian_step(h[0, 0, :], self.u_target, self.theta_step)
            return h_mod[None, None, :]
        else:
            # Prefill: Guiado continuo del token de frontera
            h_last = dispatch_riemannian_step(h[0, -1, :], self.u_target, self.theta_step)
            return mx.concatenate([h[:, :-1, :], h_last[None, None, :]], axis=1)

class AetherCollapseHead:
    """
    Envoltura de la Capa de Colapso (lm_head): modulación terminal del espacio de logits.
    """
    def __init__(self, original_lm_head):
        self.original_lm_head = original_lm_head
        self.u_terminal = None
        self.active = False

    def __getattr__(self, name):
        return getattr(self.original_lm_head, name)

    def __call__(self, h):
        if self.active and self.u_terminal is not None:
            # Confinamiento terminal antes del colapso de logits
            norm_h = mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True) + 1e-12)
            proj = mx.sum(self.u_terminal * (h / norm_h), axis=-1, keepdims=True)
            h = h + 0.1 * proj * self.u_terminal
        return self.original_lm_head(h)

class AetherEngine:
    """
    Orquestador Sistémico de AETHER: cablea las 64 capas y la capa de colapso.
    """
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.num_layers = len(self.model.language_model.model.layers)
        self.hooked_layers = []
        self.hooked_head = None
        self.settled_intent = None
        self._install_full_circuit()

    def _install_full_circuit(self):
        # 1. CABLEAR LAS 64 CAPAS DEL MODELO
        self.hooked_layers = []
        for l in range(self.num_layers):
            orig = self.model.language_model.model.layers[l]
            hook = AetherCoupledLayer(orig, layer_idx=l, num_layers=self.num_layers)
            self.model.language_model.model.layers[l] = hook
            self.hooked_layers.append(hook)

        # 2. CABLEAR LA CAPA DE COLAPSO (lm_head)
        orig_head = self.model.language_model.lm_head
        self.hooked_head = AetherCollapseHead(orig_head)
        self.model.language_model.lm_head = self.hooked_head

    def prepare_multimodal_thought(self, visual_patches, text_seed_prompt):
        # Centroide fáctico visual u_O_vis
        u_vis = mx.mean(visual_patches, axis=0)
        u_vis = u_vis / mx.sqrt(mx.sum(u_vis * u_vis) + 1e-12)

        # Semilla de intención de texto
        D = u_vis.shape[0]
        u_text = mx.random.normal(shape=(D,))
        u_text = u_text / mx.sqrt(mx.sum(u_text * u_text) + 1e-12)

        # Pensamiento Profundo continuo tau* = 32
        L_star, telemetria = run_deep_thought_settling(u_vis, u_text, tau_steps=32)
        self.settled_intent = L_star

        # CONECTAR EL CAMPO ENERGÉTICO A LAS 64 CAPAS Y AL HEAD
        for hook in self.hooked_layers:
            hook.u_target = L_star
            hook.active = True

        self.hooked_head.u_terminal = L_star
        self.hooked_head.active = True

        return telemetria
