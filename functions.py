########## Imports ##########

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from wordcloud import WordCloud





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
    rows = cmap_total//2 + 1

    plt.style.use('default')
    plt.style.use(sheet)
    fig, axes = plt.subplots(figsize=(10, rows*4), constrained_layout=True)

    for idx, cmap in enumerate(cmap_list):
        try:
            ax = plt.subplot(rows, 2, idx + 1)
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
        style = st.radio('Style Sheets', ['default', 'bmh', 'classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'seaborn', 'seaborn-colorblind', 'tableau-colorblind10'], index=0)

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

def plot_confusion_matrix(cmaps, cbar=True, reverse=False, style='default'):
    df = pd.read_csv('sample.csv').dropna()

    X = df.drop(columns=['h1n1_vaccine'])
    y = df['h1n1_vaccine']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=112221, stratify=y)

    model = LogisticRegression(random_state=112221)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    matrix = confusion_matrix(y_test, y_pred, normalize='true')
    cf = ConfusionMatrixDisplay(matrix, display_labels=['Negative', 'Positive'])

    cmap_list = cmaps_dict[cmaps]
    cmap_total = len(cmap_list)
    rows = cmap_total//2 + 1

    plt.style.use('default')
    plt.style.use(style)
    fig, axes = plt.subplots(figsize=(12, rows*5), constrained_layout=True)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    for idx, cmap in enumerate(cmap_list):
        ax = plt.subplot(rows, 2, idx + 1)

        if reverse:
            cf.plot(cmap=f'{cmap}_r', ax=ax)
        else:
            cf.plot(cmap=cmap, ax=ax)

        ax.set(title=f'{cmap}')
        
        if style in ['bmh', 'fivethirtyeight', 'ggplot', 'seaborn']:
            ax.grid()

    plt.suptitle(f'{cmaps}', fontsize=20, y=0.92)
    
    st.pyplot(fig)

def confusion_matrix_fx():
    sidebar_img(4)

    st.sidebar.write('''A confusion matrix ${C}$ is such that ${C}_{i,j}$ is equal to the number of observations known to be in group ${i}$ and predicted to be in group ${j}$ ([Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix))''')

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

    with st.sidebar.expander('More'):
        # st.code("plt.style.use('default')")
        style = st.radio('Style Sheets', ['default', 'bmh', 'classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'seaborn', 'seaborn-colorblind', 'tableau-colorblind10'], index=0)

    if st.sidebar.checkbox('Colorbar', value=True):
        colorbar = True
    else:
        colorbar = False

    if st.sidebar.checkbox('Reverse Colors', value=False):
        reverse = True
    else:
        reverse = False

    if st.button('Plot'):
        plot_confusion_matrix(colormaps, colorbar, reverse, style)

        st.markdown("<p style='font-family: Trebuchet MS;'>Code:</p>", unsafe_allow_html=True)

        code = '''plt.style.use({stylesheet})

fig, ax = plt.subplots()
matrix = confusion_matrix(y_test, y_pred, normalize='true')

cf = ConfusionMatrixDisplay(matrix, display_labels=['Negative', 'Positive'])
cf.plot(cmap={colormap}, ax=ax)

plt.show()'''
        st.code(code, language='python') 





########## Heatmap ##########

def heat_map(num_vars, cmaps, masked, annotate, cbar=True):
    pd.options.display.float_format = '{:.2f}'.format
    df = pd.read_csv('sample.csv').iloc[:, :num_vars]
    corr = df.corr()

    cmap_list = cmaps_dict[cmaps]
    cmap_total = len(cmap_list)
    rows = cmap_total//2 + 1

    plt.style.use('default')
    fig, axes = plt.subplots(figsize=(13, rows*6))

    for idx, cmap in enumerate(cmap_list):
        ax = plt.subplot(rows, 2, idx + 1)

        if masked:
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr.abs(), cmap=cmap, mask=mask, annot=annotate, cbar=cbar)
        
        else:
            sns.heatmap(corr.abs(), cmap=cmap, annot=annotate, cbar=cbar)

        ax.set(xlabel=None, ylabel=None, xticks=[], yticks=[], title=f'{cmap}')

    plt.suptitle(f'{cmaps}', fontsize=20, y=0.91)

    st.pyplot(fig)
    
def heatmap_fx():
    sidebar_img(5)

    st.sidebar.write('Plot rectangular data as a color-encoded matrix ([Seaborn](https://seaborn.pydata.org/generated/seaborn.heatmap.html))')

    vars = st.sidebar.number_input('Number of Variables', value=10, min_value=3, max_value=30, step=1)

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

    features = st.sidebar.multiselect(
     'More',
     ['Annotate', 'Colorbar', 'Mask'],
     ['Colorbar'])

    if 'Annotate' in features:
        annotate = True
    else:
        annotate = False
    if 'Colorbar' in features:
        colorbar = True
    else:
        colorbar = False
    if 'Mask' in features:
        masked = True
    else:
        masked = False

    if st.button('Plot'):
        heat_map(vars, colormaps, masked, annotate, colorbar)

        st.markdown("<p style='font-family: Trebuchet MS;'>Code:</p>", unsafe_allow_html=True)

        code = '''fig, ax = plt.subplots()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr.abs(), cmap={colormap}, mask={bool}, annot={bool}, cbar={bool})
plt.show()'''
        st.code(code, language='python')





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

wikipedia = """Matplotlib
From Wikipedia, the free encyclopedia

Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK. There is also a procedural "pylab" interface based on a state machine (like OpenGL), designed to closely resemble that of MATLAB, though its use is discouraged.[3] SciPy makes use of Matplotlib.

Matplotlib was originally written by John D. Hunter. Since then it has an active development community[4] and is distributed under a BSD-style license. Michael Droettboom was nominated as matplotlib's lead developer shortly before John Hunter's death in August 2012[5] and was further joined by Thomas Caswell.[6][7] Matplotlib is a NumFOCUS fiscally sponsored project.[8]

Matplotlib 2.0.x supports Python versions 2.7 through 3.10. Python 3 support started with Matplotlib 1.2. Matplotlib 1.4 is the last version to support Python 2.6.[9] Matplotlib has pledged not to support Python 2 past 2020 by signing the Python 3 Statement.[10]"""


def word_cloud(cmaps, reverse, num_words=30, bg='white', text_data=wikipedia):
    cmap_list = cmaps_dict[cmaps]
    cmap_total = len(cmap_list)
    rows = cmap_total//2 + 1

    plt.style.use('default')
    fig, axes = plt.subplots(figsize=(13, rows*6))

    for idx, cmap in enumerate(cmap_list):
        ax = plt.subplot(rows, 2, idx + 1)

        if reverse:
            cloud = WordCloud(max_font_size=175, width=500, height=500, max_words=num_words, colormap=f'{cmap}_r', background_color=bg, random_state=112221).generate(text_data)

        else:
            cloud = WordCloud(max_font_size=175, width=500, height=500, max_words=num_words, colormap=cmap, background_color=bg, random_state=112221).generate(text_data)

        ax.axis('off')
        plt.imshow(cloud, interpolation='bilinear')
        ax.set(title=f'{cmap}')

    plt.suptitle(f'{cmaps}', fontsize=20, y=0.91)

    st.pyplot(fig)

def wordcloud_fx():
    sidebar_img(8)

    st.sidebar.write('')

    text = st.sidebar.text_area('Words to Visualize', placeholder='''Tag cloud
From Wikipedia, the free encyclopedia
A tag cloud (also known as a word cloud, wordle or weighted list in visual design) is a visual representation of text data, which is often used to depict keyword metadata on websites, or to visualize free form text...''', help='Write your own text!')

    maxnum = st.sidebar.slider('Max Number of Words', 0, 100, 30)

    background = st.sidebar.color_picker('Background Color', '#ffffff')

    colormaps = st.sidebar.radio('Classes of Colormaps', ['Perceptually Uniform Sequential', 'Sequential', 'Sequential (2)', 'Diverging', 'Cyclic', 'Qualitative', 'Miscellaneous'])

    if st.sidebar.checkbox('Reverse Colors', value=False):
        reverse = True
    else:
        reverse = False

    st.sidebar.write('[Documentation](https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html)')

    if st.button('Plot'):
        if text is True:
            word_cloud(colormaps, reverse, maxnum, background, text)
        else:
            word_cloud(colormaps, reverse, maxnum, background)

        st.markdown("<p style='font-family: Trebuchet MS;'>Code:</p>", unsafe_allow_html=True)

        code = '''fig, ax = plt.subplots()

cloud = WordCloud(width=500, height=500, max_words={num_words}, 
    colormap={colormap}, background_color={bg_color}).generate(text)

ax.imshow(cloud, interpolation='bilinear')
ax.axis('off')

plt.show()'''
        st.code(code, language='python') 
