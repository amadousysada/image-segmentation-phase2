"""
Module de compatibilité pour les couches KerasHub
Gère les problèmes de désérialisation entre différentes versions
"""

import logging
import tensorflow as tf
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Imports des couches KerasHub originales
try:
    from keras_hub.src.models.mit.mit_layers import MixFFN as OriginalMixFFN
    from keras_hub.src.models.mit.mit_layers import OverlappingPatchingAndEmbedding as OriginalOPE
    from keras_hub.src.models.mit.mit_layers import HierarchicalTransformerEncoder as OriginalHTE
except ImportError as e:
    logger.error(f"Erreur d'import KerasHub: {e}")
    raise


def clean_config_for_hierarchical_transformer(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nettoie la configuration pour HierarchicalTransformerEncoder
    en supprimant les paramètres incompatibles
    """
    cleaned_config = config.copy()
    
    # Paramètres à supprimer car non reconnus dans la version actuelle
    params_to_remove = ['mlp', 'drop_prop']
    
    for param in params_to_remove:
        if param in cleaned_config:
            logger.warning(f"Suppression du paramètre obsolète '{param}' de la configuration")
            cleaned_config.pop(param)
    
    return cleaned_config


def clean_config_for_mixffn(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nettoie la configuration pour MixFFN
    """
    cleaned_config = config.copy()
    
    # Paramètres à supprimer
    params_to_remove = ['drop_prop']
    
    for param in params_to_remove:
        if param in cleaned_config:
            logger.warning(f"Suppression du paramètre obsolète '{param}' de MixFFN")
            cleaned_config.pop(param)
    
    return cleaned_config


class CompatibleHierarchicalTransformerEncoder(OriginalHTE):
    """
    Version compatible de HierarchicalTransformerEncoder
    qui gère gracieusement les paramètres obsolètes
    """
    
    def __init__(self, *args, **kwargs):
        # Nettoyer les kwargs avant de les passer au parent
        cleaned_kwargs = kwargs.copy()
        
        # Supprimer les paramètres non reconnus
        obsolete_params = ['mlp', 'drop_prop']
        for param in obsolete_params:
            if param in cleaned_kwargs:
                logger.warning(f"Paramètre obsolète '{param}' ignoré dans HierarchicalTransformerEncoder")
                cleaned_kwargs.pop(param)
        
        super().__init__(*args, **cleaned_kwargs)
    
    @classmethod
    def from_config(cls, config):
        """
        Créer une instance à partir d'une configuration
        en nettoyant les paramètres obsolètes
        """
        cleaned_config = clean_config_for_hierarchical_transformer(config)
        return super().from_config(cleaned_config)
    
    def get_config(self):
        """
        Retourne la configuration nettoyée
        """
        config = super().get_config()
        return clean_config_for_hierarchical_transformer(config)


class CompatibleMixFFN(OriginalMixFFN):
    """
    Version compatible de MixFFN
    """
    
    def __init__(self, *args, **kwargs):
        # Nettoyer les kwargs
        cleaned_kwargs = kwargs.copy()
        
        # Supprimer les paramètres non reconnus
        obsolete_params = ['drop_prop']
        for param in obsolete_params:
            if param in cleaned_kwargs:
                logger.warning(f"Paramètre obsolète '{param}' ignoré dans MixFFN")
                cleaned_kwargs.pop(param)
        
        super().__init__(*args, **cleaned_kwargs)
    
    @classmethod
    def from_config(cls, config):
        cleaned_config = clean_config_for_mixffn(config)
        return super().from_config(cleaned_config)
    
    def get_config(self):
        config = super().get_config()
        return clean_config_for_mixffn(config)


def get_compatible_custom_objects():
    """
    Retourne un dictionnaire d'objets personnalisés compatibles
    pour le chargement de modèles
    """
    from utils import MeanIoUArgmax
    
    return {
        "MeanIoUArgmax": MeanIoUArgmax,
        "MixFFN": CompatibleMixFFN,
        "OverlappingPatchingAndEmbedding": OriginalOPE,
        "HierarchicalTransformerEncoder": CompatibleHierarchicalTransformerEncoder,
        # Ajouter les classes originales comme fallback
        "OriginalMixFFN": OriginalMixFFN,
        "OriginalHierarchicalTransformerEncoder": OriginalHTE,
    }


def try_load_savedmodel_format(artifacts_dir: str):
    """
    Tente de charger un modèle au format SavedModel
    """
    # Chercher un dossier SavedModel dans les artifacts
    savedmodel_dirs = []
    for item in os.listdir(artifacts_dir):
        item_path = os.path.join(artifacts_dir, item)
        if os.path.isdir(item_path):
            # Vérifier si c'est un SavedModel (présence de saved_model.pb)
            if os.path.exists(os.path.join(item_path, "saved_model.pb")):
                savedmodel_dirs.append(item_path)
    
    for savedmodel_dir in savedmodel_dirs:
        try:
            logger.info(f"Tentative de chargement SavedModel depuis: {savedmodel_dir}")
            model = tf.saved_model.load(savedmodel_dir)
            logger.info("SavedModel chargé avec succès")
            return model
        except Exception as e:
            logger.warning(f"Échec chargement SavedModel: {e}")
    
    return None


def load_model_with_fallback(model_path: str, compile_model: bool = False):
    """
    Charge un modèle avec plusieurs stratégies de fallback
    """
    # Stratégie 1: Objets compatibles
    try:
        logger.info("Tentative de chargement avec objets compatibles...")
        model = tf.keras.models.load_model(
            model_path,
            compile=compile_model,
            custom_objects=get_compatible_custom_objects()
        )
        logger.info("Modèle chargé avec succès avec objets compatibles")
        return model
    except Exception as e:
        logger.warning(f"Échec avec objets compatibles: {e}")
    
    # Stratégie 2: Objets originaux
    try:
        logger.info("Tentative de chargement avec objets originaux...")
        from utils import MeanIoUArgmax
        
        original_objects = {
            "MeanIoUArgmax": MeanIoUArgmax,
            "MixFFN": OriginalMixFFN,
            "OverlappingPatchingAndEmbedding": OriginalOPE,
            "HierarchicalTransformerEncoder": OriginalHTE,
        }
        
        model = tf.keras.models.load_model(
            model_path,
            compile=compile_model,
            custom_objects=original_objects
        )
        logger.info("Modèle chargé avec succès avec objets originaux")
        return model
    except Exception as e:
        logger.warning(f"Échec avec objets originaux: {e}")
    
    # Stratégie 3: Sans compilation
    try:
        logger.info("Tentative de chargement sans compilation...")
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=get_compatible_custom_objects()
        )
        logger.info("Modèle chargé avec succès sans compilation")
        return model
    except Exception as e:
        logger.warning(f"Échec sans compilation: {e}")
    
    # Stratégie 4: Tenter de charger un SavedModel si disponible
    try:
        artifacts_dir = os.path.dirname(model_path)
        savedmodel = try_load_savedmodel_format(artifacts_dir)
        if savedmodel is not None:
            return savedmodel
    except Exception as e:
        logger.warning(f"Échec avec SavedModel: {e}")
    
    # Stratégie 5: Chargement sans custom_objects (dernier recours)
    try:
        logger.info("Tentative de chargement sans custom_objects (dernier recours)...")
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.warning("Modèle chargé sans custom_objects - certaines fonctionnalités peuvent ne pas marcher")
        return model
    except Exception as e:
        logger.error(f"Toutes les stratégies de chargement ont échoué. Dernière erreur: {e}")
        raise