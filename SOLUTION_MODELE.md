# Solutions pour le problème de désérialisation du modèle Keras

## Problème identifié

L'erreur survient lors du chargement d'un modèle Keras utilisant des couches KerasHub, spécifiquement `HierarchicalTransformerEncoder` et `MixFFN`. Le problème principal est une incompatibilité entre les paramètres utilisés lors de la sauvegarde et ceux attendus par la version actuelle de KerasHub.

### Erreur spécifique
```
ValueError: Unrecognized keyword arguments passed to HierarchicalTransformerEncoder: {'mlp': ..., 'drop_prop': 0.0}
```

## Solutions implémentées

### 1. Module de compatibilité (RECOMMANDÉ)

Un nouveau module `keras_compat.py` a été créé avec:
- Des wrappers compatibles pour les couches KerasHub
- Nettoyage automatique des paramètres obsolètes
- Stratégies de fallback multiples

**Utilisation:**
```python
from keras_compat import load_model_with_fallback
model = load_model_with_fallback(model_path, compile_model=False)
```

### 2. Versions alternatives des dépendances

Des fichiers `requirements.alternative.txt` et `Dockerfile.alternative` ont été créés avec des versions testées et compatibles:
- TensorFlow 2.15.0 (au lieu de 2.19.0)
- KerasHub 0.17.0 (au lieu de 0.21.1)

### 3. Script de diagnostic

Un script `diagnose_model.py` permet de:
- Vérifier les versions des dépendances
- Tester différentes stratégies de chargement
- Identifier les problèmes spécifiques

## Instructions pour résoudre le problème

### Option 1: Utiliser les corrections automatiques (ESSAYEZ CECI EN PREMIER)

```bash
# Reconstruire avec les corrections
docker compose up --build
```

Les corrections apportées à `main.py` et `keras_compat.py` devraient résoudre automatiquement le problème.

### Option 2: Utiliser les versions alternatives

Si l'Option 1 ne fonctionne pas:

```bash
# Construire avec les versions alternatives
docker build -f api/Dockerfile.alternative --build-arg USE_ALTERNATIVE_DEPS=true -t segmentation-api-alt ./api
```

Puis modifier `docker-compose.yml` pour utiliser cette image.

### Option 3: Diagnostic et debug

```bash
# Exécuter le diagnostic
cd api
python diagnose_model.py
```

### Option 4: Re-sauvegarder le modèle

Si toutes les options précédentes échouent, il faudra re-sauvegarder le modèle avec des versions compatibles.

## Variables d'environnement utiles

Ajoutez dans votre `.env`:

```bash
# Configuration de debugging
TF_CPP_MIN_LOG_LEVEL=2
KERAS_VERBOSE=1

# Timeout MLflow
MLFLOW_HTTP_REQUEST_TIMEOUT=6000
```

## Vérifications post-correction

Une fois le problème résolu, vous devriez voir:
```
api-1        | INFO:     Waiting for application startup.
api-1        | INFO:keras_compat:Modèle chargé avec succès avec objets compatibles
api-1        | INFO:main:Model loaded, summary :
api-1        | INFO:main:Loaded model from /tmp/.../artifacts
api-1        | INFO:     Application startup complete.
```

## Notes techniques

### Paramètres problématiques identifiés:
- `mlp`: Paramètre obsolète dans `HierarchicalTransformerEncoder`
- `drop_prop`: Paramètre obsolète, remplacé par `dropout`

### Stratégies de fallback:
1. Objets compatibles avec nettoyage automatique
2. Objets KerasHub originaux
3. Chargement sans compilation
4. Recherche de SavedModel
5. Chargement sans custom_objects (dernier recours)

## Aide supplémentaire

Si le problème persiste:
1. Vérifiez les logs détaillés avec `docker compose logs api`
2. Exécutez le script de diagnostic
3. Vérifiez que votre modèle MLflow est accessible
4. Considérez re-sauvegarder le modèle avec les versions actuelles