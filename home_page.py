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
list_paintings = [' ',
                  'Dali, Les Elephants',
                  'Giacometti, Autoportrait',
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

painting_name = st.sidebar.selectbox('Choose the Painting', list_paintings)

similarity_help = 'Larger number => Less nuances in the palette / more different colors\
   \n Lower Number => More nuance in the palette'

nb_colors = st.sidebar.slider('Number of Colors', 1, 10, 5, 1)
similarity = st.sidebar.slider('Color Separation', 30, 100, 50, 5, 
                            help=similarity_help)
# band_width = st.sidebar.slider('density', 0.1, 10., 0.5, 0.1)
# colors, counts = find_colors(image, similarity)
    # fig = show_palette(colors, counts,nb_colors)

if painting_name == ' ':
    st.image(Image.open('palette_img.png'))
    st.markdown('# Discover Palette, the app which automatically extract the main colors in paintings.')
    st.markdown("## 1)  Select a painting on the left (you may have to open the panel if you're using a smartphone).")
    st.markdown("## 2) You can  use the two sliders: \
                \n ### -  the number of colors to display \
                \n ### - the nuances of the same color to be display")

else:

    painting, palette = compute_and_show(painting_name, nb_colors,
                                        similarity, degrade=5, clip=0.2)
    # st.image()
    st.pyplot(painting)
    st.pyplot(palette)
