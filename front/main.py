import os, io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import requests

# ========== NEW: matplotlib pour charts ==========
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# --- NEW: utilitaires de mapping & colorisation du masque ---
def to_group_indices(mask_img, ignore_index=None):
    """
    Retourne un ndarray (H,W) d'indices de groupe 0..7 si possible,
    sinon None (si le masque est déjà RGB non mappable).
    """
    arr = np.array(mask_img)

    if arr.ndim == 2:
        # indices labelIds (0..33) => map vers 0..7
        if arr.max() <= 33 and np.any(arr > 7):
            grp = LABEL_TO_GROUP[arr]
        else:
            # déjà en groupes (0..7) ou autre
            grp = np.clip(arr, 0, 7).astype(np.int32)

        if ignore_index is not None:
            grp = np.where(arr == ignore_index, 7, grp)  # envoie l'ignore vers 'void'
        return grp

    elif arr.ndim == 3 and arr.shape[-1] == 3:
        # Déjà RGB → on ne tente pas de remapper: on garde tel quel
        return None

    else:
        return None


def colorize_group_mask(mask_img, ignore_index=None):
    """
    Si masque indices (0..33 ou 0..7) => colorise via GROUP_PALETTE.
    Si RGB => retourne tel quel (converti en RGB si besoin).
    """
    grp = to_group_indices(mask_img, ignore_index=ignore_index)
    if grp is None:
        return mask_img.convert("RGB") if mask_img.mode != "RGB" else mask_img

    palette = np.array(GROUP_PALETTE, dtype=np.uint8)  # (8,3)
    colored = palette[grp]                             # (H,W,3), uint8
    return Image.fromarray(colored, mode="RGB")


# =======================
# Config
# =======================
STREAMLIT_API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Segmentation — EDA + Démo", layout="wide")
st.title("Segmentation sémantique — EDA minimaliste + Démo (Cityscapes → 8 groupes)")

# =======================
# Cityscapes → 8 groupes (mapping)
# =======================
CLASS_GROUPS = {
    "flat":        ["road", "sidewalk", "parking", "rail track"],
    "human":       ["person", "rider"],
    "vehicle":     ["car", "truck", "bus", "on rails", "motorcycle", "bicycle", "caravan", "trailer"],
    "construction":["building", "wall", "fence", "guard rail", "bridge", "tunnel"],
    "object":      ["pole", "pole group", "traffic sign", "traffic light"],
    "nature":      ["vegetation", "terrain"],
    "sky":         ["sky"],
    "void":        ["unlabeled", "ego vehicle", "ground", "rectification border", "out of roi", "dynamic", "static"],
}

# Couleurs (group palette)
GROUP_PALETTE = [
    (128, 64, 128),    # 0 - flat
    (220, 20, 60),     # 1 - human
    (0, 0, 142),       # 2 - vehicle
    (70, 70, 70),      # 3 - construction
    (220, 220, 0),     # 4 - object
    (107, 142, 35),    # 5 - nature
    (70, 130, 180),    # 6 - sky
    (0, 0, 0),         # 7 - void
]
group_cmap = ListedColormap(np.array(GROUP_PALETTE) / 255.0)

DEFAULT_LEGEND = {
    0: "Flat (route/trottoir)",
    1: "Human",
    2: "Vehicle",
    3: "Construction",
    4: "Object (poteaux/panneaux)",
    5: "Nature",
    6: "Sky",
    7: "Void / Hors ROI",
}

ordered_groups = list(CLASS_GROUPS.keys())

LABEL_ID_TO_NAME = {
    0: "unlabeled", 1: "ego vehicle", 2: "rectification border", 3: "out of roi",
    4: "static", 5: "dynamic", 6: "ground", 7: "road", 8: "sidewalk", 9: "parking",
    10: "rail track", 11: "building", 12: "wall", 13: "fence", 14: "guard rail",
    15: "bridge", 16: "tunnel", 17: "pole", 18: "pole group", 19: "traffic light",
    20: "traffic sign", 21: "vegetation", 22: "terrain", 23: "sky", 24: "person",
    25: "rider", 26: "car", 27: "truck", 28: "bus", 29: "caravan", 30: "trailer",
    31: "on rails", 32: "motorcycle", 33: "bicycle",
}
NAME_TO_LABEL_ID = {v: k for k, v in sorted(LABEL_ID_TO_NAME.items())}

# table: labelId -> groupIdx
CLASS_MAP = {}
for g_idx, g_name in enumerate(ordered_groups):
    for cname in CLASS_GROUPS[g_name]:
        cid = NAME_TO_LABEL_ID.get(cname, -1)
        if cid >= 0:
            CLASS_MAP[cid] = g_idx

# mapping (longueur 256 pour sécurité) initialisé à 7 (void)
LABEL_TO_GROUP = np.full(256, 7, dtype=np.int32)
for orig_id, new_id in CLASS_MAP.items():
    LABEL_TO_GROUP[orig_id] = new_id

# couleur -> groupe (si masque RGB déjà colorisé par palette)
COLOR_TO_GROUP = {tuple(rgb): i for i, rgb in enumerate(GROUP_PALETTE)}

# =======================
# Helpers
# =======================
def load_rgb(img_file):
    return Image.open(img_file).convert("RGB")

def load_mask_any(mask_file):
    m = Image.open(mask_file)
    if m.mode in ("P", "L"):
        return m.convert("L")  # indices
    if m.mode == "RGBA":
        return m.convert("RGB")
    return m  # RGB ou autre

def equalize(img: Image.Image):
    return ImageOps.equalize(img)

def gaussian_blur(img: Image.Image, radius: int):
    return img if radius <= 0 else img.filter(ImageFilter.GaussianBlur(radius=radius))

def count_groups_from_mask(mask_img, ignore_index=None):
    """
    Retourne DataFrame: group_idx, label(str), pixels(int)
    - masque indices (H,W) en labelIds (0..33) → mappé via LABEL_TO_GROUP
    - masque indices déjà en groupes (0..7)
    - masque RGB (H,W,3) colorisé avec GROUP_PALETTE
    """
    arr = np.array(mask_img)

    if arr.ndim == 2:
        vals, counts = np.unique(arr, return_counts=True)
        # ignore
        if ignore_index is not None:
            mask = vals == ignore_index
            if mask.any():
                counts = counts.copy()
                counts[mask] = 0

        # labelIds → groupes
        if vals.max() <= 33 and np.any(vals > 7):
            g_per_label = LABEL_TO_GROUP[vals]
            group_pix = np.bincount(g_per_label, weights=counts.astype(np.int64), minlength=8)
        else:
            group_vals = np.clip(vals, 0, 7)
            group_pix = np.bincount(group_vals, weights=counts.astype(np.int64), minlength=8)

        data = []
        for g in range(8):
            p = int(group_pix[g]) if g < len(group_pix) else 0
            if p > 0:
                data.append({
                    "group_idx": g,
                    "label": DEFAULT_LEGEND.get(g, ordered_groups[g]),
                    "pixels": p
                })
        return pd.DataFrame(data)

    elif arr.ndim == 3 and arr.shape[-1] == 3:
        flat = arr.reshape(-1, 3)
        uniq, counts = np.unique(flat, axis=0, return_counts=True)
        group_pix = np.zeros(8, dtype=np.int64)
        unknown = 0
        for color, c in zip(uniq, counts):
            tup = tuple(int(x) for x in color)
            g = COLOR_TO_GROUP.get(tup, None)
            if g is None:
                unknown += int(c)
            else:
                group_pix[g] += int(c)

        data = []
        for g in range(8):
            p = int(group_pix[g])
            if p > 0:
                data.append({
                    "group_idx": g,
                    "label": DEFAULT_LEGEND.get(g, ordered_groups[g]),
                    "pixels": p
                })
        if unknown > 0:
            st.warning(f"⚠️ {unknown} pixels de couleur non reconnue (hors palette de groupes).")
        return pd.DataFrame(data)

    else:
        st.error("Format de masque non supporté.")
        return pd.DataFrame(columns=["group_idx", "label", "pixels"])

# ===== NEW: légende verticale façon show_vertical_legend =====
def show_vertical_legend(ax, palette, labels):
    ax.axis("off")
    ax.set_title("Légende", fontsize=12, pad=8)
    y0 = 1.0
    dy = 0.10
    for i, name in enumerate(labels):
        rgb = np.array(palette[i]) / 255.0
        ax.add_patch(plt.Rectangle((0.05, y0 - (i+1)*dy), 0.15, dy*0.6, color=rgb, ec="k", lw=0.5, transform=ax.transAxes))
        ax.text(0.25, y0 - (i+1)*dy + dy*0.15, f"{DEFAULT_LEGEND.get(i, name)}",
                transform=ax.transAxes, va="bottom", fontsize=10)

# ===== NEW: charts Matplotlib avec pourcentages sur le pie =====
def charts_from_counts(df_counts: pd.DataFrame):
    if df_counts.empty:
        st.info("Aucune classe/groupe détecté dans le masque.")
        return

    total = int(df_counts["pixels"].sum())
    df = df_counts.copy().sort_values("pixels", ascending=False)
    df["pourcent"] = 100.0 * df["pixels"] / max(1, total)

    # ordre & couleurs
    idxs   = df["group_idx"].tolist()
    labels = df["label"].tolist()
    pixels = df["pixels"].tolist()
    colors = [group_cmap(i) for i in idxs]

    # Figure 1: Barplot
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4), dpi=120)
    ax_bar.bar(labels, pixels, color=colors, edgecolor="black", linewidth=0.5)
    ax_bar.set_title("Pixels par groupe")
    ax_bar.set_ylabel("Pixels")
    ax_bar.set_xticklabels(labels, rotation=30, ha="right")
    fig_bar.tight_layout()

    # Figure 2: Pie + légende verticale à part
    fig_pie, (ax_pie, ax_leg) = plt.subplots(1, 2, figsize=(9, 5), dpi=120, gridspec_kw={"width_ratios":[3,1]})
    ax_pie.pie(
        df["pourcent"].values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"color": "white", "fontsize": 10}
    )
    ax_pie.axis("equal")
    ax_pie.set_title("Répartition des groupes dans le masque")

    # légende verticale (toutes les classes dans l'ordre)
    show_vertical_legend(ax_leg, GROUP_PALETTE, ordered_groups)
    fig_pie.tight_layout()

    # Render
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_bar, clear_figure=True)
    with c2:
        st.pyplot(fig_pie, clear_figure=True)

# =======================
# Onglets
# =======================
tab_eda, tab_demo = st.tabs(["🔎 EDA (upload image + masque)", "🧪 Démo segmentation (API)"])

# -----------------------
# TAB 1 — EDA simplifiée
# -----------------------
with tab_eda:
    st.subheader("Analyse exploratoire minimale (Cityscapes → 8 groupes)")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        img_file = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="eda_img")
    with col_up2:
        mask_file = st.file_uploader("Masque (PNG/JPG — indices 0..33 ou palette RGB)", type=["png", "jpg", "jpeg"], key="eda_mask")

    st.markdown("**Transformations (sur l’image)**")
    col_t1, col_t2 = st.columns([1,1])
    with col_t1:
        do_eq  = st.checkbox("Equalization", value=True)
    with col_t2:
        blur_r = st.slider("Floutage (rayon)", 0, 15, 3, step=1)

    ignore_str = st.text_input("Indice à ignorer (optionnel, ex: 255)", value="")
    IGNORE_INDEX = int(ignore_str) if ignore_str.strip().isdigit() else None

    if img_file and mask_file:
        # 1) chargement
        img  = load_rgb(img_file)
        mask = load_mask_any(mask_file)

        # 2) transforms image
        if do_eq:
            img = equalize(img)
        img = gaussian_blur(img, blur_r)

        # 3) affichage image + masque
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Image (transformée)")
            st.image(img, use_container_width=True)
        with c2:
            st.caption("Masque (colorisé par groupes)")
            mask_color = colorize_group_mask(mask, ignore_index=IGNORE_INDEX)
            st.image(mask_color, use_container_width=True)

        # 4) comptage par groupe via mapping
        df_counts = count_groups_from_mask(mask, ignore_index=IGNORE_INDEX)

        # 5) charts (camembert avec % + légende verticale)
        st.markdown("### Comptage des pixels par **groupe**")
        charts_from_counts(df_counts)

        with st.expander("Voir le tableau brut"):
            st.dataframe(df_counts, use_container_width=True)
    else:
        st.info("➡️ Charge une image **et** son masque pour lancer l’EDA.")

# -----------------------
# TAB 2 — Démo segmentation
# -----------------------
with tab_demo:
    st.subheader("Appel API de segmentation")
    uploaded_file = st.file_uploader("Choisissez une image PNG/JPG", type=["png", "jpg", "jpeg"], key="demo_img")

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.caption("Image d’entrée")
            st.image(input_image, use_container_width=True)

        if st.button("Segmenter", type="primary"):
            with st.spinner("Appel à l’API …"):
                try:
                    files = {"picture": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {"color_mode": True}
                    resp = requests.post(f"{STREAMLIT_API_URL}/segment/", files=files, params=params, timeout=60)
                    resp.raise_for_status()
                    mask_image = Image.open(io.BytesIO(resp.content))
                except requests.exceptions.HTTPError as errh:
                    st.error(f"Erreur HTTP : {errh}")
                    st.stop()
                except requests.exceptions.RequestException as err:
                    st.error(f"Erreur réseau : {err}")
                    st.stop()
                except Exception as e:
                    st.error(f"Erreur inattendue : {e}")
                    st.stop()

            with col2:
                st.caption("Masque prédit")
                st.image(mask_image, use_container_width=True, clamp=True)

            with col3:
                st.caption("Légende des groupes")
                for i, name in enumerate(ordered_groups):
                    r, g, b = GROUP_PALETTE[i]
                    st.markdown(
                        f"<span style='display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});margin-right:6px;border:1px solid #aaa;'></span> "
                        f"**{DEFAULT_LEGEND[i]}**",
                        unsafe_allow_html=True
                    )

    st.info("ℹ️ L’API attend `picture` en multipart/form-data et retourne un PNG (masque indexé ou colorisé).")