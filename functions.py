########## Imports ##########

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

########## Classes ##########

Perceptually_Uniform_Sequential = ['viridis', 'plasma', 'inferno', 'magma', 'cividis'] # 5
Sequential = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'] # 18
Sequential_2 = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'] # 16
Diverging = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'] # 12
Cyclic = ['twilight', 'twilight_shifted', 'hsv'] # 3
Qualitative = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'] # 12
Miscellaneous = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'turbo', 'nipy_spectral', 'gist_ncar'] # 'jet' 17

cmaps_dict = {'Perceptually Uniform Sequential': Perceptually_Uniform_Sequential, 
    'Sequential': Sequential, 
    'Sequential (2)': Sequential_2, 
    'Diverging': Diverging, 
    'Cyclic': Cyclic, 
    'Qualitative': Qualitative, 
    'Miscellaneous': Miscellaneous}


def sidebar_img(n):
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.image(f"icons/{n}.png")
    with col2:
        st.write("")
    with col3:
        st.write("")

########## Bar Plot ##########

def bar_plot(num_bars, cmaps, sheet):
    df = pd.DataFrame(np.random.randint(0,50,size=(num_bars, 4)), columns=list('ABCD'))

    cmap_list = cmaps_dict[cmaps]
    cmap_total = len(cmap_list)
    x = cmap_total//2 + 1

    plt.style.use(sheet)
    fig, axes = plt.subplots(figsize=(10, x*4), constrained_layout=True)

    for idx, cmap in enumerate(cmap_list):
        try:
            ax = plt.subplot(x, 2, idx + 1)
            sns.barplot(x='A', y='B', data=df, ax=ax, palette=cmap, ci=None)
            ax.set(xlabel=None, ylabel=None, xticks=[], yticks=[], title=f'{cmap}')
        except:
            pass

    plt.suptitle(f'{cmaps}', fontsize=20)
    
    st.pyplot(fig)

def bar_fx():
    sidebar_img(1)

    st.sidebar.write('A bar plot represents an estimate of central tendency for a numeric variable with the height of each rectangle ([Seaborn](https://seaborn.pydata.org/generated/seaborn.barplot.html))')
    
    bars = st.sidebar.number_input('Number of Bars', value=15, min_value=2, max_value=30, step=1)

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

    with st.sidebar.expander('More'):
        # st.code("plt.style.use('default')")
        style = st.radio('Style Sheets', ['default', 'bmh', 'classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'seaborn', 'seaborn-colorblind', 'tableau-colorblind10'])

    if st.button('Plot'):
        bar_plot(bars, colormaps, style)

        st.markdown("<p style='font-family: Trebuchet MS;'>Code:</p>", unsafe_allow_html=True)

        code = '''plt.style.use({stylesheet})
fig, ax = plt.subplots()
sns.barplot(x, y, data, ax=ax, palette={colormap}, ci=None)
plt.show()'''
        st.code(code, language='python')

########## Clustering ##########

def clustering():
    sidebar_img(2)

    st.sidebar.write('A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable ([Seaborn](https://seaborn.pydata.org/generated/seaborn.countplot.html))')

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

########## Count Plot ##########

def count_plot():
    sidebar_img(3)

    st.sidebar.write('A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable ([Seaborn](https://seaborn.pydata.org/generated/seaborn.countplot.html))')

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

########## Confusion Matrix ##########

def confusion_matrix():
    sidebar_img(4)

    st.sidebar.write('''A confusion matrix ${C}$ is such that ${C}_{i,j}$ is equal to the number of observations known to be in group ${i}$ and predicted to be in group ${j}$ ([Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix))''')


    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

########## Heatmap ##########

def heatmap():
    sidebar_img(5)

    st.sidebar.write('Plot rectangular data as a color-encoded matrix ([Seaborn](https://seaborn.pydata.org/generated/seaborn.heatmap.html))')

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

########## Histogram ##########

def histogram():
    sidebar_img(6)

    st.sidebar.write('A histogram is a classic visualization tool that represents the distribution of one or more variables by counting the number of observations that fall within disrete bins ([Seaborn](https://seaborn.pydata.org/generated/seaborn.histplot.html))')

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

########## Scatter Plot ##########

def scatter_plot():
    sidebar_img(7)

    st.sidebar.write('The scatter plot depicts the joint distribution of two variables using a cloud of points, where each point represents an observation in the dataset ([Seaborn](https://seaborn.pydata.org/tutorial/relational.html#relational-tutorial))')

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

########## Word Cloud ##########

def word_cloud():
    sidebar_img(8)

    st.sidebar.write('')

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])