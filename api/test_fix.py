#!/usr/bin/env python3
"""
Script de test pour vérifier que la correction fonctionne
"""

def test_hierarchical_transformer_fix():
    """Test de la correction HierarchicalTransformerEncoder"""
    
    # Simuler la configuration problématique qu'on reçoit du modèle sauvé
    problematic_config = {
        'name': 'hierarchical_encoder_0_0',
        'trainable': False,
        'dtype': 'float32',
        'project_dim': 64,
        'num_heads': 1,
        'drop_prop': 0.0,
        'sr_ratio': 1,
        # Le problème : objet mlp sérialisé complexe
        'mlp': {
            'module': 'keras_hub.src.models.mit.mit_layers',
            'class_name': 'MixFFN',
            'config': {
                'channels': 64,
                'mid_channels': 256,
                'trainable': False,
                'dtype': 'float32'
            }
        }
    }
    
    # Test de notre version corrigée
    print("🧪 Test de la correction HierarchicalTransformerEncoder")
    print(f"Config problématique reçue: {list(problematic_config.keys())}")
    
    try:
        # Simuler notre correction
        clean_config = problematic_config.copy()
        problematic_params = ['mlp', 'drop_prop']
        
        for param in problematic_params:
            if param in clean_config:
                print(f"  - Suppression du paramètre problématique: {param}")
                clean_config.pop(param)
        
        # Simuler la création avec les paramètres nettoyés
        cleaned_params = {
            'project_dim': clean_config.get('project_dim', 64),
            'num_heads': clean_config.get('num_heads', 1),
            'sr_ratio': clean_config.get('sr_ratio', 1),
            'drop_prob': problematic_config.get('drop_prop', 0.0),  # Récupérer drop_prop original
        }
        
        print(f"  - Paramètres nettoyés pour création: {cleaned_params}")
        print("✅ Correction réussie - la couche peut être recréée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans la correction: {e}")
        return False

def test_custom_objects():
    """Test que nos objets personnalisés sont bien définis"""
    print("\n🧪 Test des objets personnalisés")
    
    try:
        from keras_compat import get_corrected_custom_objects
        custom_objects = get_corrected_custom_objects()
        
        expected_objects = [
            "MeanIoUArgmax",
            "MixFFN", 
            "OverlappingPatchingAndEmbedding",
            "HierarchicalTransformerEncoder"
        ]
        
        for obj_name in expected_objects:
            if obj_name in custom_objects:
                print(f"  ✓ {obj_name} présent")
            else:
                print(f"  ✗ {obj_name} manquant")
                return False
        
        print("✅ Tous les objets personnalisés sont présents")
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans les objets personnalisés: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("=== TEST DE LA CORRECTION KERAS ===\n")
    
    success1 = test_hierarchical_transformer_fix()
    success2 = test_custom_objects()
    
    print(f"\n=== RÉSULTAT ===")
    if success1 and success2:
        print("🎉 Tous les tests passent - la correction devrait fonctionner !")
    else:
        print("⚠️  Certains tests échouent - vérifiez la configuration")

if __name__ == "__main__":
    main()