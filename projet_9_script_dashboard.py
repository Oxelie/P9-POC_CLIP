import os
import random
import time
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib

# Configuration de la page
st.set_page_config(page_title="Dashboard de Classification Mutlimodale avec CLIP", layout="wide")

# Fixer la seed pour la reproductibilité
seed_value = 1802
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Charger le modèle CLIP et le preprocess
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
max_length = 77
model.context_length = max_length

@st.cache_data
def load_data():
    # Charger les données
    return pd.read_csv("streamlit_dataset.csv")

data = load_data()

# dictionnaire de correspondance entre les catégories et les labels
label_to_category= {}
for label in data.label.values :
    label_to_category[label] = data[data.label == label].categ_0.values[0]
    

# Charger le classificateur entraîné
classifier = joblib.load("clip_classif_model_a.pkl")  

@st.cache_data
# Fonction pour faire des prédictions
def predict(image_path, description):
    start_time = time.time()
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([description], context_length=model.context_length, truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    combined_features = torch.cat((image_features, text_features), dim=1).cpu().numpy()
    prediction = classifier.predict(combined_features)
    end_time = time.time()
    duration = end_time - start_time
    return prediction[0], duration


# Interface utilisateur avec Streamlit
st.title("Dashboard de Classification Mutlimodale avec CLIP")

# Sélection de l'article
article_options = data['product_name'].tolist()  
article_index = st.selectbox("Sélectionnez un article", options=article_options)

# Afficher l'image et la description correspondantes
if article_index is not None:
    selected_article = data[data['product_name'] == article_index].iloc[0]  
    image_path = selected_article['reshaped_image_path']
    description = selected_article['description']
    # Afficher l'image avec une taille réduite
    image = Image.open(image_path)
    st.write(f"Image de l'article sélectionné:")
    st.image(image, caption="Image sélectionnée", width=300)  
    st.write(f"Description de l'article sélectionné:")
    st.write(f"{description}")

    # Bouton pour faire la prédiction
    st.write(f"**Prédire la catégorie avec l'image et la description de l'article en utilisant CLIP**")
    if st.button("Prédire la catégorie"):
        prediction, duration = predict(image_path, description)
        pred_category = label_to_category[prediction]
        st.write(f"Catégorie prédite")
        st.write(f"{pred_category}")
        minutes, seconds = divmod(duration, 60)
        duration_str = f"{int(minutes)} min {int(seconds)} sec"
        st.write(f"Temps de calcul de la prédiction : {duration_str}")
        

# # Prétraiter les images et les descriptions
# images = [preprocess(Image.open(image_path)) for image_path in data['reshaped_image_path']] # colonne à modfier ?
# max_length = 77
# model.context_length = max_length
# texts = clip.tokenize(data['description'], context_length=model.context_length, truncate=True).squeeze(0)

# # Empiler les images en un seul tenseur
# image_input = torch.stack(images).to(device)

# # Encoder les images et les descriptions
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(texts)
    
# # Normaliser les caractéristiques
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)

# # Combiner les caractéristiques des images et des textes
# combined_features = torch.cat((image_features, text_features), dim=1).cpu().numpy()

# # # Fonction pour faire des prédictions
# # def predict(image_path, description):
# #     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
# #     text = clip.tokenize([description]).to(device)

# #     with torch.no_grad():
# #         image_features = model.encode_image(image)
# #         text_features = model.encode_text(text)

# #     image_features /= image_features.norm(dim=-1, keepdim=True)
# #     text_features /= text_features.norm(dim=-1, keepdim=True)

# #     similarity = (image_features @ text_features.T).item()
# #     return similarity

# # Entraîner le classificateur
# classifier = LogisticRegression(max_iter=1000)
# classifier.fit(X_train, y_train)

# # Fonction pour faire des prédictions
# def predict(image_path, description):
#     start_time = time.time()
#     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#     text = clip.tokenize([description], context_length=model.context_length, truncate=True).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)

#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     combined_features = torch.cat((image_features, text_features), dim=1).cpu().numpy()
#     prediction = classifier.predict(combined_features)
#     end_time = time.time()
#     duration = end_time - start_time
#     return prediction[0], duration

# # Interface utilisateur avec Streamlit
# st.title("Dashboard de Prédiction CLIP")

# # Sélection de l'image
# image_path = st.selectbox("Sélectionnez une image", data['reshaped_image_path'].tolist())

# # Saisie de la description
# description = st.text_input("Entrez une description")

# # Bouton pour faire la prédiction
# if st.button("Prédire"):
#     if image_path and description:
#         prediction, duration = predict(image_path, description)
#         st.image(Image.open(image_path), caption="Image sélectionnée")
#         st.write(f"Description : {description}")
#         st.write(f"Catégorie prédite : {prediction}")
#         minutes, seconds = divmod(duration, 60)
#         duration_str = f"{int(minutes)} min {int(seconds)} sec"
#         st.write(f"Temps de calcul : {duration_str}")
#     else:
#         st.write("Veuillez sélectionner une image et entrer une description.")
