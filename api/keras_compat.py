"""
Module de compatibilité simplifié pour les couches KerasHub
Corrige le problème de sérialisation de HierarchicalTransformerEncoder
"""

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

# Imports des couches KerasHub originales
from keras_hub.src.models.mit.mit_layers import MixFFN as OriginalMixFFN
from keras_hub.src.models.mit.mit_layers import OverlappingPatchingAndEmbedding as OriginalOPE
from keras_hub.src.models.mit.mit_layers import HierarchicalTransformerEncoder as OriginalHTE


class HierarchicalTransformerEncoder(OriginalHTE):
    """
    Version corrigée de HierarchicalTransformerEncoder
    qui gère correctement la sérialisation/désérialisation
    """
    
    @classmethod
    def from_config(cls, config):
        """
        Méthode from_config corrigée qui ignore les paramètres problématiques
        """
        # Créer une copie de la config
        clean_config = config.copy()
        
        # Supprimer les paramètres qui causent des problèmes
        problematic_params = ['mlp', 'drop_prop']
        for param in problematic_params:
            if param in clean_config:
                logger.info(f"Suppression du paramètre problématique '{param}' lors de la désérialisation")
                clean_config.pop(param)
        
        # Créer l'instance avec la config nettoyée
        # On utilise les paramètres de base pour recréer la couche
        return cls(
            project_dim=clean_config.get('project_dim', 64),
            num_heads=clean_config.get('num_heads', 1),
            sr_ratio=clean_config.get('sr_ratio', 1),
            drop_prob=clean_config.get('drop_prop', 0.0),  # utilise drop_prop si présent
            **{k: v for k, v in clean_config.items() 
               if k not in ['project_dim', 'num_heads', 'sr_ratio', 'drop_prop', 'mlp']}
        )


def get_corrected_custom_objects():
    """
    Retourne les objets personnalisés avec la version corrigée
    """
    from utils import MeanIoUArgmax
    
    return {
        "MeanIoUArgmax": MeanIoUArgmax,
        "MixFFN": OriginalMixFFN,
        "OverlappingPatchingAndEmbedding": OriginalOPE,
        "HierarchicalTransformerEncoder": HierarchicalTransformerEncoder,  # Version corrigée
    }


def load_model_simple(model_path: str, compile_model: bool = False):
    """
    Chargement simple avec la correction ciblée
    """
    try:
        logger.info("Chargement du modèle avec correction HierarchicalTransformerEncoder...")
        model = tf.keras.models.load_model(
            model_path,
            compile=compile_model,
            custom_objects=get_corrected_custom_objects()
        )
        logger.info("✓ Modèle chargé avec succès")
        return model
    except Exception as e:
        logger.error(f"✗ Échec du chargement: {e}")
        raise