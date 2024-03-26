import streamlit as st
from PIL import Image
from utils import compute_and_show
# find_colors, show_palette
import numpy as np


st.set_page_config(
    page_title="Home page",
    page_icon="👋",
    layout="centered")

# st.write("## Welcome to the Palette")

# st.sidebar.success("Select the type of analysis you want to explore")

# st.markdown(
#     """
#     Select on the left panel what you want to explore:
#     - With 📈 forecast, you will be able to explore the Euclid Blending Forecast, with interactive histograms regarding the various blending parameters and thresholds

#     - With 👁 visualisation, you will explore examples of the different cases of blended situations.
#     """
# )
list_paintings = ['Giacometti, Autoportrait',
                  'Hirst, Fantasia Blossom',
                  'Klimt, The Kiss',
                  'Kupka, Mme Kupka among Verticals',
                  'Monet, Les Coquelicots',
                  'Sisley, Forge à Marly Leroy',
                  'Toulouse-Lautrec, Rousse',
                  'Turner, The Burning of the Houses of Lords and Commons',
                  'Van-Gogh, Nuit Etoilee',
                  'Van-Gogh, Nuit Etoilee Au Dessus Du Rhone',
                  'Kandinsky, Composition 8']

painting_name = st.sidebar.selectbox('Chose the Painting', list_paintings)


# image = np.asarray(image)[:, :, :3]


similarity_help = 'A larger number group together more similar colors, \
allowing to display more of the color spectrum. \
    A lower number will show more nuance in the main colors'

nb_colors = st.sidebar.slider('Colors number', 1, 10, 5, 1)
similarity = st.sidebar.slider('Similarity', 20, 100, 50, 5, 
                               help=similarity_help)
# band_width = st.sidebar.slider('density', 0.1, 10., 0.5, 0.1)
# colors, counts = find_colors(image, similarity)
# fig = show_palette(colors, counts,nb_colors)
painting, palette = compute_and_show(painting_name, nb_colors,
                                     similarity, 5, 0.2)
# st.image()
st.pyplot(painting)
st.pyplot(palette)
