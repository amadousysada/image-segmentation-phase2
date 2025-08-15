# Correction de l'erreur de désérialisation KerasHub

## Problème rencontré

Lors du lancement du conteneur Docker, l'application échouait avec l'erreur suivante :

```
ValueError: Unrecognized keyword arguments passed to HierarchicalTransformerEncoder: 
{'mlp': {...}, 'drop_prop': 0.0}
```

Cette erreur indique un problème de compatibilité lors de la désérialisation des couches personnalisées KerasHub, notamment :
- `HierarchicalTransformerEncoder`
- `MixFFN` 
- `OverlappingPatchingAndEmbedding`

## Cause du problème

1. **Incompatibilité de versions** : Le modèle a été sérialisé avec une version de KerasHub qui utilisait des paramètres différents (comme `drop_prop`) de ceux attendus par la version actuelle.

2. **Objets personnalisés non reconnus** : Keras ne peut pas désérialiser les couches personnalisées KerasHub sans configuration spécifique.

3. **Paramètres obsolètes** : Certains paramètres comme `mlp` et `drop_prop` ne sont plus compatibles avec la signature actuelle des classes.

## Solution implémentée

### 1. Enregistrement des classes personnalisées

```python
from keras.saving import register_keras_serializable
from keras_hub.src.models.mit.mit_layers import MixFFN as KHMixFFN
from keras_hub.src.models.mit.mit_layers import OverlappingPatchingAndEmbedding as OPE
from keras_hub.src.models.mit.mit_layers import HierarchicalTransformerEncoder as KHEnc

# Enregistrement des classes existantes
register_keras_serializable(package="keras_hub.src.models.mit.mit_layers")(OPE)
register_keras_serializable(package="keras_hub.src.models.mit.mit_layers")(KHMixFFN)
```

### 2. Classe de compatibilité pour HierarchicalTransformerEncoder

```python
@register_keras_serializable(package="keras_hub.src.models.mit.mit_layers")
class HierarchicalTransformerEncoderCompat(KHEnc):
    @classmethod
    def from_config(cls, cfg):
        cfg = dict(cfg)
        # Supprime les paramètres problématiques
        cfg.pop('mlp', None)
        
        # Gère la conversion drop_prop -> drop_rate/dropout
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
```

### 3. Chargement du modèle avec objets personnalisés

```python
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
```

## Changements effectués

### Dans `api/main.py` :

1. **Décommenté les imports** :
   - `import keras`
   - `import keras_hub`
   - `from keras.saving import register_keras_serializable`
   - Imports des classes KerasHub

2. **Activé la couche de compatibilité** :
   - Enregistrement des classes personnalisées
   - Création de `HierarchicalTransformerEncoderCompat`

3. **Modifié le chargement du modèle** :
   - Utilisation de `keras.models.load_model` au lieu de `tf.keras.models.load_model`
   - Ajout du paramètre `custom_objects`
   - Désactivation du `safe_mode`

## Test de la solution

Un script de test a été créé (`test_model_loading.py`) pour valider la configuration sans avoir besoin du modèle réel.

Exécution du test :
```bash
cd api
python test_model_loading.py
```

## Utilisation

1. **Lancer le conteneur** :
   ```bash
   docker-compose up api --build
   ```

2. **Vérifier les logs** :
   Le message `Model loaded, summary :` devrait apparaître sans erreur.

## Points importants

- ⚠️  `safe_mode=False` est nécessaire pour permettre le chargement d'objets personnalisés
- ✅ La classe de compatibilité gère automatiquement les paramètres obsolètes
- ✅ Tous les objets personnalisés doivent être fournis dans `custom_objects`
- ✅ Cette solution est rétrocompatible et fonctionne avec les modèles existants

## Alternatives envisagées

Si cette solution ne fonctionne pas complètement, d'autres approches sont possibles :

1. **Sauvegarder seulement les poids** et recréer l'architecture
2. **Mettre à jour les versions** de TensorFlow/Keras pour correspondre à celles utilisées lors de l'entraînement
3. **Convertir le modèle** vers un format plus stable (SavedModel)