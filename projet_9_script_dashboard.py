import pandas as pd
from PIL import Image
import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Dashboard de Classification Mutlimodale avec CLIP", layout="wide")


@st.cache_data
def load_data():
    # Charger les données
    return pd.read_csv("dashboard_results_with_barplots.csv")

data = load_data()

# Interface utilisateur avec Streamlit
st.title("Dashboard de Classification Mutlimodale avec CLIP")

# Injecter du CSS pour modifier la police de la selectbox
st.markdown(
    """
    <style>
    .stSelectbox label {
        font-size: 50px;
        font-weight: bold;
        --font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sélection de l'article
article_names = data['product_name'].tolist()  
article_index = st.selectbox("**Sélectionnez un article parmi une sélection aléatoire :**", options=article_names)

# Afficher l'image et la description correspondantes
if article_index is not None:
    selected_article = data[data['product_name'] == article_index].iloc[0]
    image_path = selected_article['reshaped_image_path']
    description = selected_article['description']
    true_category = selected_article['true_categ_0']
    predicted_category = selected_article['predicted_category']
    heatmap_image_path = selected_article['sampled_heatmap_image_path']
    highlighted_prompt = selected_article['highlighted_prompt']
    barplot_image_path = selected_article['barplot_image_path']


    # Afficher l'image avec une taille réduite
    image = Image.open(image_path)
    st.write(f"**Image de l'article sélectionné:**")
    st.image(image, width=300)
    st.write(f"**Description de l'article sélectionné:**")
    st.write(f"{description}")

    # Bouton pour afficher la prédiction
    st.write(" ")
    st.write('<p style="font-size:24px; font-weight:bold;">Approche Multimodale avec CLIP - Résultats</p>', unsafe_allow_html=True)
    st.write('<p style="font-size:18px;">Prédire la catégorie avec l\'image et la description de l\'article</p>', unsafe_allow_html=True)
    if st.button("**Afficher la Prédiction**"):
        st.write(f'<p style="font-size:18px;">Catégorie prédite : {predicted_category} </p>', unsafe_allow_html=True)
        st.write(f'<p style="font-size:18px;">Catégorie réelle : {true_category} </p>', unsafe_allow_html=True)
        st.write(" ")
        
        st.write('<p style="font-size:22px;font-weight:bold;">CLIP - Explications de la décision</p>', unsafe_allow_html=True)

        # Afficher l'image avec les features importances en heatmap
        st.write('<p style="font-size:18px;">Features importances sur l\'image :</p>', unsafe_allow_html=True)
        heatmap_image = Image.open(heatmap_image_path)
        st.image(heatmap_image, width=300)
        st.write(" ")

        # Afficher le highlighted_prompt
        st.write('<p style="font-size:18px;">Features importance sur le texte (mots en gras) :</p>', unsafe_allow_html=True)
        st.write(f"{highlighted_prompt}")
        st.write(" ")
        
        # Afficher l'image du barplot
        st.write(f"**Barplot des probabilités des catégories:**")
        barplot_image = Image.open(barplot_image_path)
        st.image(barplot_image, width=700)
        st.write(" ")
        st.write(" ")
        
              
# # Afficher le contenu du notebook en HTML
# st.write(" ")
# st.write(" ")
# st.write(" ")
# st.write("### Notebook d'analyses et de comparaison des modélisations")
# with open("projet_9_test_CLIP.html", "r", encoding="utf-8") as f:
#     html_content = f.read()
# components.html(html_content, height=800, scrolling=True)



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
