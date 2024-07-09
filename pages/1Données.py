import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import seaborn as sns


config.interaction = True


st.set_page_config(page_title="Données", page_icon="📈")
st.markdown("# Base à analyser")
st.sidebar.header("Données")
tab1, tab2, tab3 = st.tabs(["Base", "Statistiques", "Boxplot"])

uploaded = tab1.file_uploader("Choisissez la base", type=["xlsx"])

if 'uploaded' not in st.session_state:
    st.session_state.uploaded = None

if 'data' not in st.session_state:
    st.session_state.data = None


with tab1:
    st.header('Vue de la base')
    if uploaded is not None:
        st.session_state.uploaded = uploaded
        st.session_state.data = pd.read_excel(uploaded)

    if st.session_state.data is not None:
        tab1.write(st.session_state.data)

    if 'fac' not in st.session_state:
        st.session_state.fac = []

    if 'variable_dependante' not in st.session_state:
        st.session_state.variable_dependante = None
    if st.session_state.data is not None:
        st.session_state.variable_dependante = tab1.selectbox("Sélectionner la variable étudiée", st.session_state.data.select_dtypes(include=['int64', 'float64']).columns, index=st.session_state.data.select_dtypes(include=['int64', 'float64']).columns.get_loc(st.session_state.variable_dependante) if st.session_state.variable_dependante is not None else None)
        config.variable_dependante = st.session_state.variable_dependante
    config.data = st.session_state.data

    if 'facteurs' not in st.session_state:
        st.session_state.facteurs = 1

    if st.session_state.data is not None:
        if len(config.data.select_dtypes(include=['object','category']).columns) == 1:
            st.session_state.facteurs = 1
        elif len(config.data.select_dtypes(include=['object','category']).columns) >= 2:
            tab1.options = tab1.selectbox("Quel ANOVA voulez-vous faire?", ["1 facteur", "2 facteurs"], index=st.session_state.facteurs-1)
            st.session_state.facteurs = int(tab1.options.split()[0])
    config.facteurs = st.session_state.facteurs

    if 'potentiels_facteurs' not in st.session_state:
        st.session_state.potentiels_facteurs = []

    if st.session_state.uploaded is not None:
        st.session_state.potentiels_facteurs = [col for col in config.data.select_dtypes(include=['object','category']).columns if config.data[col].nunique() > 1]
        config.potentiels_facteurs = st.session_state.potentiels_facteurs



    if st.session_state.uploaded is not None and len(config.potentiels_facteurs) == 1:
        config.fac = config.potentiels_facteurs
    elif st.session_state.uploaded is not None and len(config.potentiels_facteurs) == 0:
        tab1.markdown("Veuillez choisir une base où l'une des variables peut être un facteur")
    elif st.session_state.uploaded is not None and config.facteurs == 1 and list(config.data.select_dtypes(include=['object','category']).columns) == config.potentiels_facteurs:
        st.session_state.fac = [tab1.selectbox(
            "Veuillez choisir un facteur",
            st.session_state.potentiels_facteurs,
            index=(st.session_state.potentiels_facteurs.index(st.session_state.fac[0]) if st.session_state.fac else 0)
        )]
        config.fac = st.session_state.fac
    elif st.session_state.uploaded is not None and config.facteurs == 2:
        st.session_state.fac = tab1.multiselect("Veuillez choisir les facteurs",st.session_state.potentiels_facteurs,max_selections=2, default=st.session_state.fac if len(st.session_state.fac)==2 else None)
        config.fac = tab1.options

with tab2:

    st.header(" Statistiques descriptives")
    if st.session_state.data is not None and len(st.session_state.fac) != 0 and st.session_state.variable_dependante is not None:
        try:
            tab2.write(st.session_state.data.groupby(st.session_state.fac)[st.session_state.variable_dependante].mean())
        except:
            tab2.write()
        for i in st.session_state.fac:
            st.session_state.data[i] = st.session_state.data[i].astype('category')
        config.data = st.session_state.data

    if st.session_state.data is not None:
        variables_quantitatives = st.session_state.data.select_dtypes(include=[np.number])
        tab2.write(st.session_state.data.select_dtypes(include=[np.number]).describe().transpose())





with tab3:
    # Check if data and variables are selected
    if st.session_state.data is not None and st.session_state.variable_dependante is not None and st.session_state.facteurs == 2:

        # Define custom colors
        st.header("Boxplot de la variable à étudier")
        unique_varietes = st.session_state.data[st.session_state.fac[1]].unique() if st.session_state.fac else None
        colors = ['blue', 'orange', 'white', 'red'][:len(unique_varietes)] if st.session_state.fac else None
        palette = dict(zip(unique_varietes, colors)) if st.session_state.fac else None
        # Create the plot
        plt.figure(figsize=(10, 6))
            # Boxplot
        if st.session_state.fac:
            sns.boxplot(
                data=st.session_state.data,
                x=st.session_state.fac[0],
                y=st.session_state.variable_dependante,
                hue=st.session_state.fac[1],
                palette=palette,
                dodge=True,
                fliersize=0
            )
        # Jittered points
        if st.session_state.fac:
            sns.stripplot(
                data=st.session_state.data,
                x=st.session_state.fac[0],
                y=st.session_state.variable_dependante,
                hue=st.session_state.fac[1],
                palette=palette,
                dodge=True,
                jitter=True,
                size=5,
                alpha=0.7
            )
        # Add mean lines
        if st.session_state.fac:
            for i, treat in enumerate(st.session_state.data[st.session_state.fac[0]].unique()):
                for j, variete in enumerate(st.session_state.data[st.session_state.fac[1]].unique()):
                    subset = st.session_state.data[
                        (st.session_state.data[st.session_state.fac[0]] == treat) &
                        (st.session_state.data[st.session_state.fac[1]] == variete)
                    ]
                    mean = subset[st.session_state.variable_dependante].mean()
                    plt.plot([i-0.2, i+0.2], [mean, mean], color=palette[variete], linestyle='-', linewidth=1.5)

            plt.ylabel(st.session_state.variable_dependante)
            plt.legend(title=st.session_state.fac[1], bbox_to_anchor=(1.05, 1), loc='upper left')
            sns.despine()

        # Display plot in Streamlit
        st.pyplot(plt)
