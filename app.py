########## Imports ##########

import streamlit as st
import base64
from st_clickable_images import clickable_images
import functions as fx

st.set_page_config(
    page_title="Palette Picker",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

########## Header ##########

st.markdown('<h2 style="text-align:center;"><a href="/" style="text-align:center; color:black; font-family:Trebuchet MS; text-decoration:None" target="_self">The Palette Picker</a></h2>', unsafe_allow_html=True)

st.write("<p style='text-align: center; font-family: Helvetica;'>Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.</p>", unsafe_allow_html=True)

st.caption("<p style='text-align: center; font-family: Helvetica;'> Click on a Plot </p>", unsafe_allow_html=True)

icons = []
for i in range(1,9):
    icons.append(f'icons/{i}.png')

images = []
for file in icons:
    with open(file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        images.append(f"data:image/jpeg;base64,{encoded}")

img_idx = clickable_images(
    images,
    titles=['Bar Plot', 'Clustering', 'Count Plot', 'Confusion Matrix', 'Heatmap', 'Histogram', 'Scatter Plot', 'Word Cloud'],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "40px"},
)

plot_dict = {0: 'Bar Plot',
 1: 'Clustering',
 2: 'Count Plot',
 3: 'Confusion Matrix',
 4: 'Heatmap',
 5: 'Histogram',
 6: 'Scatter Plot',
 7: 'Word Cloud'}

fx_dict = []

########## Plot ##########

if img_idx < 0:
    plot = st.selectbox(
        '',
        ('Pick a plot...', 'Bar Plot', 'Clustering', 'Count Plot', 'Confusion Matrix', 'Heatmap', 'Histogram', 'Scatter Plot', 'Word Cloud'))
    if plot == 'Bar Plot':
        fx.bar_fx()
    if plot == 'Clustering':
        fx.clustering()
    if plot == 'Count Plot':
        fx.count_plot()
    if plot == 'Confusion Matrix':
        fx.confusion_matrix_fx()
    if plot == 'Heatmap':
        fx.heatmap_fx()
    if plot == 'Histogram':
        fx.histogram()
    if plot == 'Scatter Plot':
        fx.scatter_plot()
    if plot == 'Word Cloud':
        fx.wordcloud_fx()

else:
    st.subheader(f"{plot_dict[img_idx]}")
    if img_idx == 0:
        fx.bar_fx()
    if img_idx == 1:
        fx.clustering()
    if img_idx == 2:
        fx.count_plot()
    if img_idx == 3:
        fx.confusion_matrix_fx()
    if img_idx == 4:
        fx.heatmap_fx()
    if img_idx == 5:
        fx.histogram()
    if img_idx == 6:
        fx.scatter_plot()
    if img_idx == 7:
        fx.wordcloud_fx()


########## Palette ##########

# st.subheader('Colormaps')
st.markdown('<h5>Colormaps</h5>', unsafe_allow_html=True)
st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.")

colormaps = []
for i in range(1,8):
    colormaps.append(f'images/{i}.png')

classes = []
for file in colormaps:
    with open(file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        classes.append(f"data:image/jpeg;base64,{encoded}")

cmap_idx = clickable_images(
    classes,
    titles=['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "100px"},
)

# st.caption("<p style='text-align: center; font-family: Helvetica;'> Click on a Palette </p>", unsafe_allow_html=True)

if cmap_idx > -1:
    st.image(f'colormaps/class{cmap_idx}.png')
    with st.expander('Lightness'):
        st.image(f'colormaps/light{cmap_idx}.png')

st.write('[Matplotlib Cheatsheets](https://matplotlib.org/cheatsheets/cheatsheets.pdf)')

########## Footer ##########

st.caption("<p style='text-align: center; font-family: Helvetica;'><br><br><br><br><br>The Palette Picker is created by <a href='http://czarinaluna.com/'>Czarina Luna</a>.<br><br>Any addition is highly encouraged. <a href='https://github.com/czarinagluna/plt-palette-picker/issues'>Fill an issue</a> on Github or contact me on <a href='https://www.linkedin.com/in/czarinaluna/'>Linkedin</a>.</p>", unsafe_allow_html=True)


