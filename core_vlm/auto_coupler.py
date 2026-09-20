import os, json, math
import mlx.core as mx

class HookedDecoderLayer:
    def __init__(self, original_layer, operator_fn):
        self.original_layer = original_layer
        self.operator_fn = operator_fn

    def __getattr__(self, name):
        return getattr(self.original_layer, name)

    def __call__(self, x, mask=None, cache=None, position_ids=None, position_embeddings=None):
        h = self.original_layer(x, mask=mask, cache=cache, position_ids=position_ids, position_embeddings=position_embeddings)
        h_last = h[0, -1, :]
        h_last_mod = self.operator_fn(h_last)
        return mx.concatenate([h[:, :-1, :], h_last_mod[None, None, :]], axis=1)

class AutoCoupler:
    def __init__(self, model_dir):
        self.model_dir = os.path.expanduser(model_dir)
        self.config_path = os.path.join(self.model_dir, "config.json")
        self.config = self._load_config()
        self.params = self._derive_scaling_laws()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ No se encontró config.json en {self.model_dir}")
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _derive_scaling_laws(self):
        cfg = self.config
        text_cfg = cfg.get("text_config", cfg)
        d_model = text_cfg.get("hidden_size", text_cfg.get("d_model", cfg.get("hidden_size", 5120)))
        num_layers = text_cfg.get("num_hidden_layers", text_cfg.get("n_layer", cfg.get("num_hidden_layers", 64)))
        vocab_size = text_cfg.get("vocab_size", cfg.get("vocab_size", 248320))
        arch_type = cfg.get("model_type", "transformer")

        kappa_0 = 1.0 / math.sqrt(float(d_model))
        alpha_eml = kappa_0 / 2.0
        dt_canonical = 1.0 / float(min(64, num_layers))
        lambda_dissipation = kappa_0 * (dt_canonical ** 2)

        l_start = int(math.floor(0.25 * num_layers))
        l_end   = int(math.floor(0.60 * num_layers))

        return {
            "d_model": d_model,
            "num_layers": num_layers,
            "vocab_size": vocab_size,
            "model_type": arch_type,
            "kappa_0": kappa_0,
            "alpha_eml": alpha_eml,
            "dt_canonical": dt_canonical,
            "lambda_dissipation": lambda_dissipation,
            "locus_band_LM": (l_start, l_end),
            "rank_R": min(32, d_model // 32)
        }

    def get_causal_layer_index(self):
        l_start, l_end = self.params["locus_band_LM"]
        return (l_start + l_end) // 2

    def attach(self, model, operator_fn, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.get_causal_layer_index()
        self._attached_layer_idx = layer_idx
        self._original_layer = model.language_model.model.layers[layer_idx]
        hooked = HookedDecoderLayer(self._original_layer, operator_fn)
        model.language_model.model.layers[layer_idx] = hooked
        return layer_idx

    def detach(self, model):
        if hasattr(self, "_attached_layer_idx") and hasattr(self, "_original_layer"):
            model.language_model.model.layers[self._attached_layer_idx] = self._original_layer
            del self._attached_layer_idx
            del self._original_layer
