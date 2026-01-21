import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from collections import defaultdict
import pickle

# -------------------- Tema --------------------
st.set_page_config(
    page_title="Outfit Recommendation System",
    page_icon="👗",
    layout="centered"
)

# -------------------- Model --------------------
@st.cache_resource
def load_model():
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_features():
    return np.load("item_features.npy", allow_pickle=True).item()

@st.cache_resource
def load_graph():
    with open("outfit_graph.gpickle", "rb") as f:
        return pickle.load(f)

features_dict = load_features()
G = load_graph()

# -------------------- Yardımcı Fonksiyonlar --------------------
def resize_and_pad(image, size=(224, 224), color=(255, 255, 255)):
    old_size = image.size
    ratio = float(size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", size, color)
    paste_position = ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2)
    new_image.paste(image, paste_position)
    return new_image

def extract_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor).squeeze().numpy()
    return vec

def recommend_outfits_return_similars(new_image_path, features_dict, G, top_k_items=5, top_k_outfits=5):
    new_vec = extract_vector(new_image_path)
    similarities = [(item_id, cosine_similarity([new_vec], [vec])[0][0]) for item_id, vec in features_dict.items()]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_items = similarities[:top_k_items]

    outfit_scores = defaultdict(float)
    for item_id, sim in top_items:
        for neighbor in G.neighbors(item_id):
            if G.nodes[neighbor]["type"] == "outfit":
                outfit_scores[neighbor] += sim

    ranked_outfits = sorted(outfit_scores.items(), key=lambda x: x[1], reverse=True)
    return top_items, [outfit for outfit, _ in ranked_outfits[:top_k_outfits]]

def show_combination_comparison(new_image_path, target_outfit_id, features_dict, base_folder):
    outfit_path = os.path.join(base_folder, target_outfit_id)
    original_items = [
        os.path.join(outfit_path, img)
        for img in os.listdir(outfit_path)
        if img.endswith(".jpg")
    ]

    new_vec = extract_vector(new_image_path)
    item_similarities = []
    for img_path in original_items:
        item_id = os.path.basename(img_path).split(".")[0]
        item_vec = features_dict.get(item_id)
        if item_vec is not None:
            sim = cosine_similarity([new_vec], [item_vec])[0][0]
            item_similarities.append((img_path, sim))

    item_similarities.sort(key=lambda x: x[1], reverse=True)
    replaced_item_path, _ = item_similarities[0]
    new_combination = [img for img in original_items if img != replaced_item_path]
    new_combination.append(new_image_path)

    st.markdown("<h5>Orijinal Kombin</h5>", unsafe_allow_html=True)
    cols = st.columns(len(original_items))
    for i, img_path in enumerate(original_items):
        img = resize_and_pad(Image.open(img_path).convert("RGB"))
        cols[i].image(img, use_container_width=True)

    st.markdown("<h5 style='margin-top:2rem;'>Yeni Ürünle Güncellenmiş Kombin</h5>", unsafe_allow_html=True)
    cols_new = st.columns(len(new_combination))
    for i, img_path in enumerate(new_combination):
        img = resize_and_pad(Image.open(img_path).convert("RGB"))
        caption = "Yeni Ürün" if img_path == new_image_path else ""
        cols_new[i].image(img, caption=caption, use_container_width=True)

# -------------------- Arayüz --------------------
st.markdown("<h2 style='color:#222;'>👗 Outfit Recommendation System</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;'>Yeni bir ürün resmi yükleyin ve sistemin önerdiği kombinleri inceleyin.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Ürün resmini yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp_uploaded.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = resize_and_pad(Image.open("temp_uploaded.jpg").convert("RGB"))
    st.image(img, caption="Yüklenen Ürün", width=300)

    base_folder = r"C:\\Users\\semer\\Desktop\\Resimler1"
    similar_items, suggested_outfits = recommend_outfits_return_similars("temp_uploaded.jpg", features_dict, G)

    st.subheader("Önerilen Kombinler")
    choice = st.radio("Bir kombin seçin:", suggested_outfits, horizontal=False)

    for outfit in suggested_outfits:
        with st.expander(f"Kombin ID: {outfit}", expanded=False):
            show_combination_comparison("temp_uploaded.jpg", outfit, features_dict, base_folder)

    # Renkli buton CSS
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #e63946;
            color: white;
            height: 3em;
            width: 100%;
            border-radius:10px;
            border: none;
            font-size:16px;
        }
        div.stButton > button:first-child:hover {
            background-color: #d62828;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("Seçimi Kaydet"):
        df = pd.DataFrame([[uploaded_file.name, choice]], columns=["Uploaded_Image", "Selected_Outfit"])
        if os.path.exists("selections.csv"):
            df.to_csv("selections.csv", mode='a', header=False, index=False)
        else:
            df.to_csv("selections.csv", index=False)
        st.success("Seçim başarıyla kaydedildi ✅")

# -------------------- TÜBİTAK LOGOSU ve Footer --------------------
st.markdown("---")

tubitak_logo_path = r"C:\Users\semer\Desktop\OutfitGUI\tubitak_logo.png"
if os.path.exists(tubitak_logo_path):
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(tubitak_logo_path, width=150)
    st.markdown("""
        <p style='color:gray; font-size:13px; margin-top:8px;'>
            Bu proje TÜBİTAK 2209-A programı tarafından desteklenmektedir.
        </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="text-align:center;">
            <p style='color:gray; font-size:13px;'>
                Bu proje TÜBİTAK 2209-A programı tarafından desteklenmektedir.
            </p>
        </div>
    """, unsafe_allow_html=True)
