from __future__ import annotations

import logging

import mlflow
import os, tempfile

from contextlib import asynccontextmanager
from fastapi import FastAPI

from routes import router
from settings import get_settings
import tensorflow as tf
import keras
from utils import MeanIoUArgmax, Model, dice_loss
import keras_hub
# from keras_hub.models import MiTBackbone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf = get_settings()

MLFLOW_TRACKING_URI = conf.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# --- compat_keras_hub_mit.py (ou directement en haut de ton main) ---
import inspect
from keras.saving import register_keras_serializable

# Importe les vraies classes depuis l'API publique installée
from keras_hub.src.models.mit.mit_layers import MixFFN as KHMixFFN
from keras_hub.src.models.mit.mit_layers import OverlappingPatchingAndEmbedding as OPE
from keras_hub.src.models.mit.mit_layers import HierarchicalTransformerEncoder as KHEnc

# Enregistre des alias sous le *chemin sérialisé* de ton .keras
register_keras_serializable(package="keras_hub.src.models.mit.mit_layers")(OPE)
register_keras_serializable(package="keras_hub.src.models.mit.mit_layers")(KHMixFFN)

@register_keras_serializable(package="keras_hub.src.models.mit.mit_layers")
class HierarchicalTransformerEncoderCompat(KHEnc):
    @classmethod
    def from_config(cls, cfg):
        cfg = dict(cfg)
        # Supprime les paramètres qui posent problème
        cfg.pop('mlp', None)
        # Gère le paramètre drop_prop
        if "drop_prop" in cfg:
            sig = inspect.signature(KHEnc.__init__)
            params = sig.parameters
            if "drop_rate" in params: 
                cfg["drop_rate"] = cfg.pop("drop_prop")
            elif "dropout" in params: 
                cfg["dropout"] = cfg.pop("drop_prop")
            elif "dropout_rate" in params: 
                cfg["dropout_rate"] = cfg.pop("drop_prop")
            else: 
                cfg.pop("drop_prop", None)
        return cls(**cfg)

async def load_model():
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "6000")
    client = mlflow.MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = client.download_artifacts(
                 conf.RUN_ID,
                 "artifacts",
                 dst_path=temp_dir
             )
            keras_model_path = os.path.join(model_path, "mit_segformer_model.keras")
            logger.info(f"Loading model from: {keras_model_path}")
            
            # Charger le modèle avec Keras 3.x avec les objets personnalisés
            model = keras.models.load_model(
                keras_model_path,
                compile=False,
                safe_mode=False,  # Nécessaire pour les objets personnalisés
                custom_objects={
                    "MeanIoUArgmax": MeanIoUArgmax,
                    "dice_loss": dice_loss,
                    "HierarchicalTransformerEncoder": HierarchicalTransformerEncoderCompat,
                    "MixFFN": KHMixFFN,
                    "OverlappingPatchingAndEmbedding": OPE,
                }
            )
            logger.info("Model loaded, summary :")
            model.summary()
            Model().set_model(model)
            logger.info("Loaded model from %s", model_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed loading model %s: %s", model_path, exc)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await load_model()
    except Exception as exc:
        logger.error("Failed loading model during startup: %s", exc)
        raise
    yield
    logger.info("Application shutdown complete.")

app = FastAPI(title="Segmentation API", version="1.0.0", lifespan=lifespan)

# Include all routes
app.include_router(router=router)