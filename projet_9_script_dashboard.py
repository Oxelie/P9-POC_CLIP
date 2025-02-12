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
st.markdown("""
            Bienvenue sur ce tableau de bord intéractif ! 
            
               Ici, vous pourrez :
               - Choisir un article parmi une sélection aléatoire,
               - Découvrir les résultats de la prédiction avec CLIP,
               - Analyser les décisions du modèle.
               """)
# mise en forme du texte avec CSS
st.markdown(
    """
    <style>
    .stSelectbox label {
        font-size: 50px;
        font-weight: bold;
        font-style: italic;
    }
    .rounded-frame {
        border: 2px solid #000000;
        border-radius: 15px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True)


# Sélection de l'article
st.write(" ")
st.write('<p style="font-size:24px; font-weight:bold;">Prédictions avec CLIP</p>', unsafe_allow_html=True)
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
    # st.markdown('</div>', unsafe_allow_html=True)
    
    st.write(f"**Description de l'article sélectionné:**")
    st.write(f"{description}")
 

    # Bouton pour afficher la prédiction
    st.write(" ")
    st.write('<p style="font-size:20px; font-weight:bold;">Résultats</p>', unsafe_allow_html=True)
    st.write("Prédire la catégorie avec l\'image et la description de l\'article")
    if st.button("**Afficher la Prédiction**"):
        st.write(f'<p style="font-size:18px;font-weight:bold;">Catégorie prédite : {predicted_category} </p>', unsafe_allow_html=True)
        st.write(f'<p style="font-size:18px;">Catégorie réelle : {true_category} </p>', unsafe_allow_html=True)
        st.write(" ")
        
        st.write('<p style="font-size:22px;font-weight:bold;">CLIP - Explications de la décision</p>', unsafe_allow_html=True)

        # Afficher l'image avec les features importances en heatmap
        st.write('<p style="font-size:18px;font-weight:bold;">Features importances sur l\'image :</p>', unsafe_allow_html=True)
        heatmap_image = Image.open(heatmap_image_path)
        st.image(heatmap_image, width=300)
        st.write(" ")

        # Afficher le highlighted_prompt
        st.write('<p style="font-size:18px;font-weight:bold;">Features importance sur le texte (mots en gras) :</p>', unsafe_allow_html=True)
        st.write(f"{highlighted_prompt}")
        st.write(" ")
        
        # Afficher l'image du barplot
        st.write('<p style="font-size:18px;font-weight:bold;">Barplot des probabilités des catégories :</p>', unsafe_allow_html=True)
        barplot_image = Image.open(barplot_image_path)
        st.image(barplot_image, width=700)
        st.write(" ")
        st.write(" ")
        
  