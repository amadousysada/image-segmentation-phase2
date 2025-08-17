import logging

logger = logging.getLogger(__name__)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.saving import register_keras_serializable
from keras.saving import serialize_keras_object, deserialize_keras_object

# Couches originales Keras Hub
from keras_hub.src.models.mit.mit_layers import (
    MixFFN as OriginalMixFFN,
    OverlappingPatchingAndEmbedding as OriginalOPE,
    HierarchicalTransformerEncoder as OriginalHTE,
)

@register_keras_serializable(
    package="keras_hub.src.models.mit.mit_layers",
    name="HierarchicalTransformerEncoder"
)
class HTEFixed(OriginalHTE):
    """
    Shim de compatibilité pour HierarchicalTransformerEncoder.
    - Sérialise le sous-layer `mlp`.
    - Pendant la désérialisation, ne passe pas `mlp` au constructeur;
      le reconstruit puis l'attache à l'objet.
    - Normalise `drop_prop` -> `drop_prob` et simplifie `dtype`.
    """

    def get_config(self):
        base = super().get_config()
        cfg = dict(base)

        # Inclure `mlp` s'il existe et que c'est bien un layer
        mlp = getattr(self, "mlp", None)
        if isinstance(mlp, keras.layers.Layer):
            cfg["mlp"] = serialize_keras_object(mlp)

        # (Optionnel) simplifier dtype si c'est une DTypePolicy sérialisée
        if isinstance(cfg.get("dtype"), dict):
            cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")

        # Harmoniser le nom de l'arg si jamais c'est "drop_prop" côté ancien artefact
        if "drop_prop" in cfg and "drop_prob" not in cfg:
            cfg["drop_prob"] = cfg.pop("drop_prop")

        return cfg

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)

        # Nettoyage des champs qui posent souci
        if isinstance(cfg.get("dtype"), dict):
            cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")
        if "drop_prop" in cfg and "drop_prob" not in cfg:
            cfg["drop_prob"] = cfg.pop("drop_prop")

        # Extraire le sous-layer sérialisé `mlp`, ne PAS le passer au constructeur
        mlp_cfg = cfg.pop("mlp", None)

        # Construire l'objet sans `mlp`
        obj = OriginalHTE(**cfg)

        # Désérialiser et rattacher `mlp` après construction si fourni
        if mlp_cfg is not None:
            mlp_layer = deserialize_keras_object(
                mlp_cfg,
                custom_objects={"MixFFN": OriginalMixFFN}
            )
            obj.mlp = mlp_layer

        return obj


def _flatten_dtype(cfg):
    if isinstance(cfg.get("dtype"), dict):
        cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")
    return cfg

# --- Fix MixFFN: retirer les kwargs inattendus (trainable/dtype/build_config) ---
@register_keras_serializable(
    package="keras_hub.src.models.mit.mit_layers",
    name="MixFFN",
)
class MixFFN_Fixed(OriginalMixFFN):
    def __init__(self, **kwargs):
        if "dtype" in kwargs:
            kwargs.pop("dtype")
            super().__init__(**kwargs)
    def get_config(self):
        base = super().get_config()
        cfg = dict(base)

        # (Optionnel) simplifier dtype si c'est une DTypePolicy sérialisée
        if isinstance(cfg.get("dtype"), dict):
            cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")

        return cfg

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        # certains builds sérialisent ces champs, mais l'__init__ de MixFFN ne les accepte pas
        cfg.pop("trainable", None)
        #cfg.pop("build_config", None)
        cfg.pop("dtype", None)
        # selon l'implémentation, "name" est OK (Layer), on le garde
        return OriginalMixFFN(**cfg)
def get_corrected_custom_objects():
    """
    Retourne les objets personnalisés avec la version corrigée
    """
    from utils import MeanIoUArgmax

    return {
        "MeanIoUArgmax": MeanIoUArgmax,
        "MixFFN": MixFFN_Fixed,
        "OverlappingPatchingAndEmbedding": OriginalOPE,
        "HierarchicalTransformerEncoder": HTEFixed,  # Version corrigée
    }


def load_model_simple(model_path: str, cust_obj: dict, compile_model: bool = False):
    """
    Chargement simple avec la correction ciblée
    """
    cust_obj.update(**get_corrected_custom_objects())
    try:
        logger.info("Chargement du modèle avec correction HierarchicalTransformerEncoder...")
        model = tf.keras.models.load_model(
            model_path,
            compile=compile_model,
            safe_mode=True,
            custom_objects=cust_obj
        )
        logger.info("✓ Modèle chargé avec succès")
        return model
    except Exception as e:
        logger.error(f"✗ Échec du chargement: {e}")
        raise