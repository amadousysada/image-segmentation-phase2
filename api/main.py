from __future__ import annotations

import logging
from urllib.parse import urlparse

import mlflow
import os, tempfile, zipfile

from contextlib import asynccontextmanager
from fastapi import FastAPI

from routes import router
from settings import get_settings
import tensorflow as tf
import keras
from utils import MeanIoUArgmax, Model, dice_loss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf = get_settings()

MLFLOW_TRACKING_URI = conf.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

async def load_model():
    client = mlflow.MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = client.download_artifacts(
                conf.RUN_ID,
                "model",
                dst_path=temp_dir
            )
            keras_model_path = os.path.join(model_path, "data", "model.keras")
            logger.info(f"Loading model from: {keras_model_path}")

            # Charger le modèle avec Keras 3.x
            model = tf.keras.models.load_model(keras_model_path, custom_objects={"MeanIoUArgmax": MeanIoUArgmax, "dice_loss": dice_loss})
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