from __future__ import annotations

import logging

import mlflow
import os, tempfile

from contextlib import asynccontextmanager
from fastapi import FastAPI

from routes import router
from settings import get_settings
import tensorflow as tf

from utils import MeanIoUArgmax, Model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf = get_settings()

MLFLOW_TRACKING_URI = conf.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

from keras_hub.src.models.mit.mit_layers import MixFFN as KHMixFFN
from keras_hub.src.models.mit.mit_layers import OverlappingPatchingAndEmbedding as OPE
from keras_hub.src.models.mit.mit_layers import HierarchicalTransformerEncoder as KHEnc

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
            model = tf.keras.models.load_model(
                keras_model_path,
                compile=False,
                custom_objects={
                    "MeanIoUArgmax": MeanIoUArgmax,
                    "MixFFN": KHMixFFN,
                    "OverlappingPatchingAndEmbedding": OPE,
                    "HierarchicalTransformerEncoder": KHEnc,
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