import os
import io
import requests
import streamlit as st
from PIL import Image
import numpy as np

# Config
STREAMLIT_API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000").rstrip('/')

st.set_page_config(page_title="Segmentation Demo", layout="centered")
st.title("Segmentation sémantique d'image")

# 1. Upload d'image
uploaded_file = st.file_uploader("Choisissez une image PNG ou JPG", type=["png", "jpg", "jpeg"])

# Option pour le mode couleur
#color_mode = st.checkbox("Utiliser des couleurs pour le masque (plus facile à distinguer)", value=True)

if uploaded_file is not None:
    # Affiche l'image d'entrée

    input_image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader("Image d'entrée")
        st.image(input_image, use_container_width=True)

    # 2. Bouton pour lancer la requête
    if st.button("Segmenter"):
        with st.spinner("Appel à l’API de segmentation…"):
            try:
                # On renvoie le fichier sous form-data (clé 'file' ou autre selon votre API)
                files = {"picture": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"color_mode": True}
                resp = requests.post(f"{STREAMLIT_API_URL}/segment/", files=files, params=params, timeout=60)
                resp.raise_for_status()
                # 3. Lecture du mask PNG retourné
                mask_bytes = resp.content
                mask_image = Image.open(io.BytesIO(mask_bytes))
            except requests.exceptions.HTTPError as errh:
                st.error(f"Erreur HTTP : {errh}")
                st.stop()
            except requests.exceptions.RequestException as err:
                st.error(f"Erreur réseau : {err}")
                st.stop()
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
                st.stop()
        # Informations de débogage
        mask_array = np.array(mask_image)
        unique_values = np.unique(mask_array)
        orig_width, orig_height = input_image.size
        with col2:
            st.image(mask_image, width=orig_width, clamp=True)
        
        with col3:
            #st.write("**Informations du masque:**")
            #st.write(f"📏 Dimensions: {mask_array.shape}")
            #st.write(f"🎨 Mode: Couleurs RGB")
            #st.write(f"🖼️ Classes détectées: {len(np.unique(mask_array.flatten())) if len(mask_array.shape) == 3 else len(unique_values)}")

            # Légende des couleurs pour le mode couleur (correspondant au GROUP_PALETTE du notebook)
            st.write("**Légende des couleurs:**")
            color_legend = [
                "🟣 Flat (route, trottoir)",
                "🔴 Human (personne, cycliste)",
                "🔵 Vehicle (voiture, camion)",
                "⚫  Construction (bâtiment, mur)",
                "🟡 Object (poteau, panneau)",
                "🟢 Nature (végétation, terrain)",
                "🩵 Sky (ciel) - Bleu ciel",
                "🖤  Void (non labellisé, hors ROI)"
            ]
            for legend in color_legend:
                st.write(f"{legend}")