# Solution Simple pour le problème de désérialisation du modèle Keras

## Problème identifié

L'erreur survient lors du chargement d'un modèle Keras utilisant `HierarchicalTransformerEncoder` de KerasHub. Le problème est que cette classe sauvegarde l'objet `mlp` complet dans sa méthode `get_config()`, mais cette sérialisation complexe échoue lors de la désérialisation.

### Erreur spécifique
```
ValueError: Unrecognized keyword arguments passed to HierarchicalTransformerEncoder: {'mlp': ..., 'drop_prop': 0.0}
```

## Solution implémentée (SIMPLE ET EFFICACE)

J'ai créé une version corrigée de `HierarchicalTransformerEncoder` qui surcharge uniquement la méthode `from_config()` pour ignorer les paramètres problématiques et recréer la couche correctement.

### Code de la solution (`keras_compat.py`)

```python
class HierarchicalTransformerEncoder(OriginalHTE):
    @classmethod
    def from_config(cls, config):
        # Nettoyer la config des paramètres problématiques
        clean_config = config.copy()
        problematic_params = ['mlp', 'drop_prop']
        for param in problematic_params:
            if param in clean_config:
                clean_config.pop(param)
        
        # Recréer la couche avec les bons paramètres
        return cls(
            project_dim=clean_config.get('project_dim', 64),
            num_heads=clean_config.get('num_heads', 1),
            sr_ratio=clean_config.get('sr_ratio', 1),
            drop_prob=clean_config.get('drop_prop', 0.0),
            **{k: v for k, v in clean_config.items() 
               if k not in ['project_dim', 'num_heads', 'sr_ratio', 'drop_prop', 'mlp']}
        )
```

## Instructions pour résoudre le problème

### Étape unique: Reconstruire l'image Docker

```bash
docker compose up --build
```

C'est tout ! La correction automatique devrait résoudre le problème.

## Pourquoi cette solution fonctionne

1. **Problème racine**: `HierarchicalTransformerEncoder.get_config()` sauvegarde l'objet `mlp` complet via `keras.saving.serialize_keras_object(self.mlp)`, mais cette sérialisation complexe échoue.

2. **Solution ciblée**: Au lieu d'essayer de désérialiser l'objet `mlp` complexe, on l'ignore complètement et on laisse la classe recréer ses propres couches internes dans son `__init__()`.

3. **Minimal et robuste**: Cette approche ne change que ce qui est nécessaire et préserve tout le reste du comportement original.

## Vérification du succès

Une fois corrigé, vous devriez voir dans les logs :
```
api-1        | INFO:keras_compat:Chargement du modèle avec correction HierarchicalTransformerEncoder...
api-1        | INFO:keras_compat:✓ Modèle chargé avec succès
api-1        | INFO:main:Model loaded, summary :
api-1        | INFO:     Application startup complete.
```

## Si le problème persiste

Si cette solution ne fonctionne pas, utilisez le script de diagnostic pour plus d'informations :

```bash
cd api
python diagnose_model.py
```

## Notes techniques

- **Pas de changement de versions** : Cette solution fonctionne avec vos versions actuelles (TensorFlow 2.19.0, KerasHub 0.21.1)
- **Correction ciblée** : Seule la désérialisation de `HierarchicalTransformerEncoder` est modifiée
- **Préservation de fonctionnalité** : Toutes les autres couches et fonctionnalités restent intactes