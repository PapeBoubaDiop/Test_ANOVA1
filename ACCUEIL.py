import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(
    page_title="ACCUEIL",
    page_icon="👋",
)
st.snow()
st.sidebar.success("Présentation")
st.write("# Welcome to the app of DIOP Papa Boubacar and OUEDRAOGO Faïçal Cheick Hamed")
st.title('Projet d\'analyse de la variance')
st.markdown(
    """
    Cette application a été programmé dans le cadre d'un projet d'Analyse de la variance (ANOVA).
    Dans cette application, vous aurez la possibilité de mettre des bases sous format excel, ... pour faire de l'ANOVA.
"""
)















#fig, ax = plt.subplots(figsize=(10, 6))
#sns.boxplot(data=data, x='Traitement', y='Long', hue='Code_Variete', 
#            palette=['blue', 'orange', 'white', 'red'], dodge=True, fliersize=0, linewidth=1, ax=ax)
#sns.stripplot(data=data, x='Traitement', y='Long', hue='Code_Variete', 
#              palette=['blue', 'orange', 'white', 'red'], dodge=True, jitter=True, size=2, alpha=0.5, ax=ax)

# Affichez le graphique dans Streamlit
#st.pyplot(fig) """