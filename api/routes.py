import logging
from typing import Dict

from fastapi import APIRouter
from fastapi import UploadFile, HTTPException, File, Query

import tensorflow as tf

from starlette.responses import Response

from utils import preprocess_image, postprocess_mask, postprocess_mask_color, Model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health-check/", summary="Healthcheck")
def root() -> Dict[str, str]:
    return {"status": "ok"}

@router.post("/segment/")
async def predict(
    *,
    picture: UploadFile = File(...),
    color_mode: bool = Query(False, description="Use color mask instead of grayscale for better visualization"),
) -> Response:
    """Return image segment (mask)."""
    if picture.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Image must be JPEG or PNG")

    model = Model().get_model()

    data = await picture.read()
    x = preprocess_image(data)
    mask_img = model.predict(x)
    logger.info(f"Predicted: {mask_img}")
    
    # Choisir entre masque couleur ou niveaux de gris
    if color_mode:
        png_bytes = postprocess_mask_color(mask_img)
        logger.info("PostProcessed mask in color mode")
    else:
        png_bytes = postprocess_mask(mask_img)
        logger.info("PostProcessed mask in grayscale mode")
        
    return Response(content=png_bytes, media_type="image/png")