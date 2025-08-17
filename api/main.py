from __future__ import annotations

import logging
import os, warnings, tempfile

# 1) Forcer CPU si pas de GPU (élimine le message CUDA 303)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2) Réduire le bruit TensorFlow côté C++ (INFO/WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 3) Masquer le warning Keras "build() was called..."
warnings.filterwarnings(
    "ignore",
    message=r"`build\(\)` was called on layer '.*'[, ].*does not have a `build\(\)` method.*",
    category=UserWarning,
    module=r"keras\.src\.layers\.layer",
)
import mlflow

from contextlib import asynccontextmanager
from fastapi import FastAPI

from keras_compat import load_model_simple
from routes import router
from settings import get_settings
import tensorflow as tf
import cloudpickle
from utils import MeanIoUArgmax, Model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf = get_settings()

MLFLOW_TRACKING_URI = conf.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

async def load_model():
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "6000")
    client = mlflow.MlflowClient()
    from importlib.metadata import version
    print("Les versions des paquets:")
    print(f"keras_hub => {version('keras_hub')}")
    print(f"keras => {version('keras')}")
    print(f"tensorflow => {version('tensorflow')}")
    print(f"tensorflow keras => {tf.keras.__version__}")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Temp dir: {temp_dir}")
            model_path = client.download_artifacts(
                 conf.RUN_ID,
                 "model",
                 dst_path=temp_dir
             )
            obj = {}
            keras_model_path = os.path.join(model_path, "data", "model.keras")
            obj_path = os.path.join(model_path, "data", "global_custom_objects.cloudpickle")
            with open(obj_path, "rb") as f:
                obj = cloudpickle.load(f)
            model = load_model_simple(keras_model_path, obj, compile_model=False)
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