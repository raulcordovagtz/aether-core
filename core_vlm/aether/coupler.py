import mlx.core as mx
from .kernels import dispatch_riemannian_step, dispatch_alu_cell
from .settling import run_deep_thought_settling

TARGET_LAYER = 24

class AetherCoupledLayer:
    def __init__(self, original_layer):
        self.original_layer = original_layer
        self.u_target = None
        self.theta_rad = 0.35
        self.active = False
        self.alu_active = False

    def __getattr__(self, name):
        return getattr(self.original_layer, name)

    def __call__(self, x, **kwargs):
        h = self.original_layer(x, **kwargs)
        if not self.active or self.u_target is None:
            return h

        if h.shape[1] == 1:
            # Decode: Paso riemanniano en silicio
            h_mod = dispatch_riemannian_step(h[0, 0, :], self.u_target, self.theta_rad)
            return h_mod[None, None, :]
        else:
            # Prefill: Modulación del último token
            h_last = dispatch_riemannian_step(h[0, -1, :], self.u_target, self.theta_rad)
            return mx.concatenate([h[:, :-1, :], h_last[None, None, :]], axis=1)

class AetherEngine:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.hooked_layer = None
        self.settled_intent = None
        self._install_hook()

    def _install_hook(self):
        orig_layer = self.model.language_model.model.layers[TARGET_LAYER]
        self.hooked_layer = AetherCoupledLayer(orig_layer)
        self.model.language_model.model.layers[TARGET_LAYER] = self.hooked_layer

    def prepare_multimodal_thought(self, visual_patches, text_seed_prompt):
        """
        Fase de Prefill: Extrae u_vis, u_text y ejecuta el Pensamiento Profundo tau* = 32.
        """
        # Centroide fáctico visual u_O_vis
        u_vis = mx.mean(visual_patches, axis=0)
        u_vis = u_vis / mx.sqrt(mx.sum(u_vis * u_vis) + 1e-12)

        # Semilla de intención de lenguaje
        D = u_vis.shape[0]
        tokens = self.processor.tokenizer.encode(text_seed_prompt)
        u_text = mx.random.normal(shape=(D,)) # Semilla ortogonal de base
        u_text = u_text / mx.sqrt(mx.sum(u_text * u_text) + 1e-12)

        # Asentamiento continuo tau* = 32
        L_star, telemetria = run_deep_thought_settling(u_vis, u_text, tau_steps=32)
        self.settled_intent = L_star

        # Conectar el timón con el estado óptimo asentado
        self.hooked_layer.u_target = L_star
        self.hooked_layer.active = True
        return telemetria
