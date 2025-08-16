#!/usr/bin/env python3
"""
Script de diagnostic pour les problèmes de chargement de modèle
"""

import logging
import os
import sys
import tensorflow as tf
import mlflow
from settings import get_settings
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_tensorflow_version():
    """Vérifier la version de TensorFlow"""
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # Vérifier si GPU est disponible
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs disponibles: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")

def check_keras_hub():
    """Vérifier KerasHub"""
    try:
        import keras_hub
        print(f"KerasHub version: {keras_hub.__version__}")
        
        # Tester l'import des couches problématiques
        from keras_hub.src.models.mit.mit_layers import MixFFN
        from keras_hub.src.models.mit.mit_layers import HierarchicalTransformerEncoder
        from keras_hub.src.models.mit.mit_layers import OverlappingPatchingAndEmbedding
        
        print("✓ Import des couches KerasHub réussi")
        
        # Vérifier les paramètres du constructeur
        import inspect
        hte_params = inspect.signature(HierarchicalTransformerEncoder.__init__).parameters
        print(f"Paramètres HierarchicalTransformerEncoder: {list(hte_params.keys())}")
        
        mixffn_params = inspect.signature(MixFFN.__init__).parameters
        print(f"Paramètres MixFFN: {list(mixffn_params.keys())}")
        
    except ImportError as e:
        print(f"✗ Erreur d'import KerasHub: {e}")
    except Exception as e:
        print(f"✗ Erreur KerasHub: {e}")

def check_mlflow_connection():
    """Vérifier la connexion MLflow"""
    try:
        conf = get_settings()
        mlflow.set_tracking_uri(conf.MLFLOW_TRACKING_URI)
        
        client = mlflow.MlflowClient()
        run = client.get_run(conf.RUN_ID)
        print(f"✓ Connexion MLflow réussie")
        print(f"  Run ID: {conf.RUN_ID}")
        print(f"  Run status: {run.info.status}")
        print(f"  Tracking URI: {conf.MLFLOW_TRACKING_URI}")
        
        # Lister les artifacts
        artifacts = client.list_artifacts(conf.RUN_ID)
        print(f"  Artifacts disponibles:")
        for artifact in artifacts:
            print(f"    - {artifact.path} ({artifact.file_size} bytes)")
            
        return True
    except Exception as e:
        print(f"✗ Erreur connexion MLflow: {e}")
        return False

def download_and_inspect_model():
    """Télécharger et inspecter le modèle"""
    try:
        conf = get_settings()
        client = mlflow.MlflowClient()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Téléchargement des artifacts dans: {temp_dir}")
            
            # Télécharger les artifacts
            model_path = client.download_artifacts(
                conf.RUN_ID,
                "artifacts",
                dst_path=temp_dir
            )
            
            print(f"Artifacts téléchargés dans: {model_path}")
            
            # Lister le contenu
            print("Contenu des artifacts:")
            for root, dirs, files in os.walk(model_path):
                level = root.replace(model_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"{subindent}{file} ({file_size} bytes)")
            
            # Chercher le fichier .keras
            keras_model_path = os.path.join(model_path, "mit_segformer_model.keras")
            if os.path.exists(keras_model_path):
                print(f"✓ Modèle Keras trouvé: {keras_model_path}")
                
                # Tenter de charger avec différentes stratégies
                test_model_loading(keras_model_path)
            else:
                print(f"✗ Modèle Keras non trouvé: {keras_model_path}")
                
    except Exception as e:
        print(f"✗ Erreur téléchargement modèle: {e}")

def test_model_loading(model_path):
    """Tester différentes stratégies de chargement"""
    print(f"\nTest de chargement du modèle: {model_path}")
    
    strategies = [
        ("Sans custom_objects", {}),
        ("Avec MeanIoUArgmax", {"MeanIoUArgmax": "utils.MeanIoUArgmax"}),
        ("Avec KerasHub original", "original"),
        ("Avec KerasHub compatible", "compatible"),
    ]
    
    for strategy_name, custom_objects in strategies:
        try:
            print(f"\n  Tentative: {strategy_name}")
            
            if custom_objects == "original":
                from keras_hub.src.models.mit.mit_layers import MixFFN
                from keras_hub.src.models.mit.mit_layers import HierarchicalTransformerEncoder
                from keras_hub.src.models.mit.mit_layers import OverlappingPatchingAndEmbedding
                from utils import MeanIoUArgmax
                
                custom_objects = {
                    "MeanIoUArgmax": MeanIoUArgmax,
                    "MixFFN": MixFFN,
                    "OverlappingPatchingAndEmbedding": OverlappingPatchingAndEmbedding,
                    "HierarchicalTransformerEncoder": HierarchicalTransformerEncoder,
                }
            elif custom_objects == "compatible":
                from keras_compat import get_compatible_custom_objects
                custom_objects = get_compatible_custom_objects()
            
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects if custom_objects else None
            )
            
            print(f"    ✓ Succès avec {strategy_name}")
            print(f"    Modèle: {type(model)}")
            print(f"    Input shape: {model.input_shape if hasattr(model, 'input_shape') else 'N/A'}")
            print(f"    Output shape: {model.output_shape if hasattr(model, 'output_shape') else 'N/A'}")
            
            return model
            
        except Exception as e:
            print(f"    ✗ Échec avec {strategy_name}: {str(e)[:100]}...")
    
    print("  ✗ Toutes les stratégies ont échoué")
    return None

def main():
    """Fonction principale de diagnostic"""
    print("=== DIAGNOSTIC DU MODÈLE ===\n")
    
    print("1. Vérification des versions:")
    check_tensorflow_version()
    print()
    
    print("2. Vérification de KerasHub:")
    check_keras_hub()
    print()
    
    print("3. Vérification de la connexion MLflow:")
    if check_mlflow_connection():
        print()
        print("4. Téléchargement et inspection du modèle:")
        download_and_inspect_model()
    
    print("\n=== FIN DU DIAGNOSTIC ===")

if __name__ == "__main__":
    main()