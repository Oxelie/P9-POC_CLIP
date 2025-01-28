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

# Charger les données
data_final = pd.read_csv("path_to_your_data.csv")  # CRÉER UN FICHIER CSV AVEC UNE PETITE SELCTION DE DONNÉES PAR CATÉGORIE

# Charger le classificateur entraîné
classifier = joblib.load("path_to_your_trained_model.pkl")  # Remplacez par le chemin de votre modèle enregistré

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
st.title("Dashboard de Prédiction CLIP")

# Sélection de l'article
article_options = data_final['title'].tolist()  # Remplacez 'title' par la colonne appropriée du nom de l'article sinon index ?
article_index = st.selectbox("Sélectionnez un article", options=article_options)

# Afficher l'image et la description correspondantes
if article_index is not None:
    selected_article = data_final[data_final['title'] == article_index].iloc[0]  # Remplacez 'title' par la colonne appropriée
    image_path = selected_article['reshaped_image_path']
    description = selected_article['description']
    st.image(Image.open(image_path), caption="Image sélectionnée")
    st.write(f"Description : {description}")

    # Bouton pour faire la prédiction
    if st.button("Prédire"):
        prediction, duration = predict(image_path, description)
        st.write(f"Catégorie prédite : {prediction}")
        minutes, seconds = divmod(duration, 60)
        duration_str = f"{int(minutes)} min {int(seconds)} sec"
        st.write(f"Temps de calcul : {duration_str}")
        

# # Prétraiter les images et les descriptions
# images = [preprocess(Image.open(image_path)) for image_path in data_final['reshaped_image_path']] # colonne à modfier ?
# max_length = 77
# model.context_length = max_length
# texts = clip.tokenize(data_final['description'], context_length=model.context_length, truncate=True).squeeze(0)

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
# image_path = st.selectbox("Sélectionnez une image", data_final['reshaped_image_path'].tolist())

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
