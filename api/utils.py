import mlflow
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
DEFAULT_LEGEND = {
    0: "🟣 Flat (route, trottoir)",
    1: "🔴 Human (personne, cycliste)",
    2: "🔵 Vehicle (voiture, camion)",
    3: "⚫  Construction (bâtiment, mur)",
    4: "🟡 Object (poteau, panneau)",
    5: "🟢 Nature (végétation, terrain)",
    6: "🩵 Sky (ciel) - Bleu ciel",
    7: "🖤  Void (non labellisé, hors ROI)"
}
class MeanIoUArgmax(tf.keras.metrics.MeanIoU):
    """Custom MeanIoU metric that applies argmax to predictions"""
    def __init__(self, num_classes, name="mean_io_u_argmax", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred : (batch, H, W, num_classes) → take the winning class
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss for semantic segmentation

    Args:
        y_true: Ground truth masks (batch_size, H, W)
        y_pred: Predicted logits (batch_size, H, W, num_classes)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice loss value
    """
    # Convert predictions to probabilities
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    # Convert ground truth to one-hot encoding
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

    # Flatten tensors
    y_true_flat = tf.reshape(y_true_one_hot, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred, [-1, num_classes])

    # Calculate Dice coefficient for each class
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)

    dice_coeff = (2. * intersection + smooth) / (union + smooth)

    # Return 1 - mean Dice coefficient as loss
    return 1 - tf.reduce_mean(dice_coeff)


class Model:
    instance = None
    initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Model, cls).__new__(cls)
        return cls.instance

    def __init__(self, model=None):
        if not self.initialized:
            self.initialized = True
            self.model: mlflow.pyfunc.PyFuncModel = model

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

def preprocess_image(file_bytes: bytes):
    img = tf.image.decode_png(file_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # shape → (1, 224, 224, 3)

    return img

def postprocess_mask(mask_logits: np.ndarray):
    """
    mask_logits: np.ndarray shape (1, H, W, C)
    Retourne les octets d'un PNG du mask (H, W, 1) en uint8.
    """
    # argmax → (1, H, W)
    mask_indices = np.argmax(mask_logits, axis=-1)
    # squeeze batch → (H, W)
    mask_indices = mask_indices[0]
    
    # Palette de couleurs optimisée pour 8 classes (0-7)
    # Chaque classe aura une valeur bien distincte en niveaux de gris
    color_map = np.array([
        0,    # Classe 0: Noir (background généralement)
        36,   # Classe 1: Gris très foncé
        73,   # Classe 2: Gris foncé
        109,  # Classe 3: Gris moyen-foncé
        146,  # Classe 4: Gris moyen
        182,  # Classe 5: Gris moyen-clair
        219,  # Classe 6: Gris clair
        255   # Classe 7: Blanc
    ])
    
    # Appliquer la palette de couleurs
    if mask_indices.max() < len(color_map):
        mask_normalized = color_map[mask_indices]
    else:
        # Fallback: normalisation linéaire si plus de 8 classes détectées
        mask_normalized = (mask_indices / mask_indices.max() * 255).astype(np.uint8)
    
    # Convertir en uint8 si ce n'est pas déjà fait
    mask_normalized = mask_normalized.astype(np.uint8)
    
    # ajout du canal → (H, W, 1)
    mask_normalized = mask_normalized[..., np.newaxis]
    # encode en PNG
    encoded = tf.io.encode_png(mask_normalized).numpy()
    return encoded

def postprocess_mask_color(mask_logits: np.ndarray):
    """
    Alternative en couleurs pour une meilleure visualisation des 8 classes.
    mask_logits: np.ndarray shape (1, H, W, C)
    Retourne les octets d'un PNG du mask (H, W, 3) en couleurs RGB.
    """
    # argmax → (1, H, W)
    mask_indices = np.argmax(mask_logits, axis=-1)
    # squeeze batch → (H, W)
    mask_indices = mask_indices[0]
    
    # Palette de couleurs RGB correspondant au GROUP_PALETTE du notebook
    color_palette = np.array([
        [128, 64, 128],   # Classe 0: flat (route, trottoir, etc.) - Violet-gris
        [220, 20, 60],    # Classe 1: human (personne, cycliste) - Rouge-crimson
        [0, 0, 142],      # Classe 2: vehicle (voiture, camion, etc.) - Bleu foncé
        [70, 70, 70],     # Classe 3: construction (bâtiment, mur, etc.) - Gris foncé
        [220, 220, 0],    # Classe 4: object (poteau, panneau, etc.) - Jaune
        [107, 142, 35],   # Classe 5: nature (végétation, terrain) - Vert olive
        [70, 130, 180],   # Classe 6: sky (ciel) - Bleu ciel
        [0, 0, 0]         # Classe 7: void (non labellisé, hors ROI) - Noir
    ], dtype=np.uint8)
    
    # Créer l'image couleur
    h, w = mask_indices.shape
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(min(len(color_palette), mask_indices.max() + 1)):
        mask_color[mask_indices == class_id] = color_palette[class_id]
    
    # encode en PNG
    encoded = tf.io.encode_png(mask_color).numpy()
    return encoded
