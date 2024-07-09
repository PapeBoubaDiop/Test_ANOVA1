import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.api import jarque_bera
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import levene

st.set_page_config(page_title="Anova")
st.markdown("# Résultats de l'ANOVA")
st.sidebar.header("Résultats")
tab1, tab2, tab3,tab4 = st.tabs(["Hypothèses","Résultats", "Interpretations","Graphique"])

if 'facteur0' not in st.session_state:
    st.session_state.facteur0 = None
if 'facteur1' not in st.session_state:
    st.session_state.facteur1 = None
if 'facteur2' not in st.session_state:
    st.session_state.facteur2 = None
if 'model' not in st.session_state:
    st.session_state.model = None

if 'interaction' not in st.session_state:
    st.session_state.interaction = None


with tab1:
    tab1.write("### Vérification des hypothèses")

    if st.session_state.facteurs == 2 and st.session_state.data is not None and len(st.session_state.fac)!=0:
        st.session_state.interaction = tab1.selectbox("Type d'analyse",['ANOVA avec interaction','ANOVA sans interaction'], index=None)
        tab1.write(st.session_state.interaction)
    if 'button_anova' not in st.session_state:
        st.session_state.button_anova = False

    if 'data' in st.session_state and len(st.session_state.fac)>0:
        if st.session_state.facteurs == 1:
            st.session_state.facteur0 = st.session_state.fac[0]
            formula = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur0})"
            st.session_state.model = ols(formula, data=st.session_state.data).fit()
            anova_lm(st.session_state.model,typ=2)

        if st.session_state.facteurs == 2:
            st.session_state.facteur1 = st.session_state.fac[0]
            st.session_state.facteur2 = st.session_state.fac[1]
            if st.session_state.interaction == "ANOVA sans interaction":
                formula3 = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur2}) + C({st.session_state.facteur1})"
                st.session_state.model = ols(formula3, data=st.session_state.data).fit()
                anova_lm(st.session_state.model,typ=2)
            else:    
                formula2 = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur2}) * C({st.session_state.facteur1})"
                st.session_state.model = ols(formula2, data=st.session_state.data).fit()
                anova_lm(st.session_state.model,typ=2)  






    with st.expander("**NORMALITE**"):
        st.header("Vérification de la normalité")
        # Histogramme des résidus
        st.write("**Histogramme des résidus**")
        fig, ax = plt.subplots()
        if st.session_state.model:
            sns.histplot(st.session_state.model.resid, kde=True, ax=ax)
        st.pyplot(fig)
        # Q-Q plot des résidus
        st.write("**Q-Q plot des résidus**")
        fig = sm.qqplot(st.session_state.model.resid, line='s')
        st.pyplot(fig)
        # Test de normalité de Shapiro-Wilk
        st.write("#### Test de Shapiro-Wilk")
        st.write("**Hypothèse nulle (H0)** : Les résidus suivent une distribution normale.")
        st.write("**Hypothèse alternative (H1)** : Les résidus ne suivent pas une distribution normale.")
        st.session_state.shapiro_test = stats.shapiro(st.session_state.model.resid)
        st.write(f"**Statistic**: {st.session_state.shapiro_test.statistic}, \n**P-value**: {st.session_state.shapiro_test.pvalue}")
        if st.session_state.shapiro_test.pvalue >= 0.05:
            st.write("**L'hypothèse de normalité est vérifiée ✔**")
        else:
            st.write("**L'hypothèse de normalité n'est pas vérifiée ❌**")
#        jb_test = jarque_bera(st.session_state.model.resid)
#        st.write(jb_test)


    with st.expander("**HOMOSCEDASTICITE**"):
        st.header("Vérification de l'homogénéité")

        # Test de Levene pour l'homogénéité des variances
        #st.session_state.groups = [st.session_state.data[st.session_state.data[st.session_state.fac[0]] == factor][response] for factor in st.session_state.data[st.session_state.fac[0]].unique()]
        #st.session_state.levene_test = levene(*st.session_state.groups)
        #st.write("**Hypothèse nulle (H0)** : Les variances des résidus sont homogènes.")
        #st.write(" **Hypothèse alternative (H1)** : Les variances des résidus ne sont pas homogènes.")
        #st.write("Test de Levene pour l'homogénéité des variances:")
        #st.write(f"Statistic: {st.session_state.levene_test.statistic}, P-value: {st.session_state.levene_test.pvalue}")
        #if st.session_state.levene_test.pvalue < 0.05:
        #    st.write("Les variance des résidus ne sont pas homogène : l'hypothèse est pas vérifée ✔")
        #else:
        #    st.session_state.levene_test.pvalue > 0.05:
        #    st.write("Les variance des résidus sont homogènes : l'hypothèse n'est pas vérifée ❌")

        # Résidus vs valeurs ajustées
        st.write("**Graphique des résidus vs valeurs ajustées**")
        st.session_state.fitted_vals = st.session_state.model.fittedvalues
        fig, ax = plt.subplots()
        ax.scatter(st.session_state.fitted_vals, st.session_state.model.resid)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Valeurs ajustées')
        ax.set_ylabel('Résidus')
        st.pyplot(fig)



    with st.expander("**AUTOCORRELATION**"):
        st.header("Vérification de l'indépendance")
        st.write("#### Test de Durbin-Watson")
        st.write("**Hypothèse nulle (H0)** : Les résidus sont indépendants.")
        st.write("**Hypothèse alternative (H1)** : Les résidus ne sont pas indépendants.")
        # Test de Durbin-Watson pour l'indépendance des résidus
        st.session_state.dw_test = durbin_watson(st.session_state.model.resid)
        st.write(f"**Statistic**: {st.session_state.dw_test}")
        if st.session_state.dw_test < 1:
            st.write("**Il y a une autocorrélation positive des résidus : l'hypothèse n'est pas vérifée ❌**")
        elif st.session_state.dw_test > 3:
            st.write("**Il y a une autocorrélation négative des résidus : l'hypothèse n'est pas vérifée ❌**")
        else:
            st.write("**Il y a pas d'autocorrélation significative des résidus : l'hypothèse est vérifée ✔**")


# Significativité des p-value
def format_pvalues_with_stars(pvalues):
    formatted_pvalues = []
    stars = []
    for p in pvalues:
        if p < 0.01:
            stars.append('***')
        elif p < 0.05:
            stars.append('**')
        elif p < 0.1:
            stars.append('*')
        else:
            stars.append('')
        
        if p < 0.001:
            formatted_pvalues.append(f"{p:.2e}")
        else:
            formatted_pvalues.append(f"{p:.3f}")
    
    return formatted_pvalues, stars



with tab2:
    if st.session_state.shapiro_test.pvalue >= 0.05:
        if st.session_state.facteurs == 2 and st.session_state.data is not None and len(st.session_state.fac)!=0:
            tab2.write(st.session_state.interaction)
        if 'button_anova' not in st.session_state:
            st.session_state.button_anova = True

        def click_button_anova():
            st.session_state.button_anova = True
        st.button('Lancer l\'ANOVA', on_click=click_button_anova)

        if st.session_state.button_anova and 'data' in st.session_state and len(st.session_state.fac)>0:
            if st.session_state.facteurs == 1:
                st.session_state.facteur0 = st.session_state.fac[0]
                formula = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur0})"
                st.session_state.model = ols(formula, data=st.session_state.data).fit()
                st.session_state.anova_df = pd.DataFrame(anova_lm(st.session_state.model,typ=2))
                formatted_pvalues, significance_stars = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])
                st.session_state.anova_df['p-value'] = formatted_pvalues
                st.session_state.anova_df['Significativité'] = significance_stars 
                st.session_state.anova_df = st.session_state.anova_df.drop(columns=['PR(>F)'])           
                tab2.write("ANOVA à un facteur")
                tab2.write(st.session_state.anova_df)
            if st.session_state.facteurs == 2:
                st.session_state.facteur1 = st.session_state.fac[0]
                st.session_state.facteur2 = st.session_state.fac[1]
                if st.session_state.interaction == "ANOVA avec interaction":    
                    formula2 = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur2}) * C({st.session_state.facteur1})"
                    st.session_state.model = ols(formula2, data=st.session_state.data).fit()
                    st.session_state.anova_df = pd.DataFrame(anova_lm(st.session_state.model,typ=2))
                    formatted_pvalues, significance_stars = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])
                    st.session_state.anova_df['p-value'] = formatted_pvalues
                    st.session_state.anova_df['Significativité'] = significance_stars 
                    st.session_state.anova_df = st.session_state.anova_df.drop(columns=['PR(>F)'])           
                    tab2.write(st.session_state.anova_df)

                
                elif st.session_state.interaction == "ANOVA sans interaction":
                    formula3 = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur2}) + C({st.session_state.facteur1})"
                    st.session_state.model = ols(formula3, data=st.session_state.data).fit()
                    st.session_state.anova_df = pd.DataFrame(anova_lm(st.session_state.model,typ=2))
                    formatted_pvalues, significance_stars = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])
                    st.session_state.anova_df['p-value'] = formatted_pvalues
                    st.session_state.anova_df['Significativité'] = significance_stars 
                    st.session_state.anova_df = st.session_state.anova_df.drop(columns=['PR(>F)'])           
                    tab2.write(st.session_state.anova_df)
    else:
        st.write("### L'hypothèse de normalité n'étant pas respectée, l'analyse de la variance n'est pas appropriée. Nous recommandons l'utilisation d'un modèle non paramétrique pour cette analyse 😊 ")
        

with tab4:
        # Function to calculate mean and confidence interval
    def summarySE(data, measurevar, groupvars):
        summary_df = data.groupby(groupvars).agg(
            N=(measurevar, 'count'),
            mean=(measurevar, 'mean'),
            std=(measurevar, 'std')
        ).reset_index()
        
        summary_df['ci'] = 1.96 * summary_df['std'] / np.sqrt(summary_df['N'])  # 95% CI
        return summary_df

    # Calculer la moyenne et les intervalles de confiance
    data_avg = summarySE(st.session_state.data, measurevar=st.session_state.variable_dependante, groupvars=st.session_state.fac)
    data_avg.to_csv("datagrp.csv", sep=";", index=False, na_rep="NA")

    # Visualisation des moyennes et des intervalles de confiance
    st.subheader("Moyenne par groupe et intervalle de confiance")
    p = sns.color_palette(["#1E90FF", "#D02090", "#FFFFFF", '#A81D35'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_avg, x=st.session_state.fac[0], y='mean', hue=st.session_state.fac[1], palette=p, marker='o', err_style="bars", ci=None)

    # Ajout des barres d'erreur manuellement
    for name, group in data_avg.groupby(st.session_state.fac[1]):
        plt.errorbar(group[st.session_state.fac[0]], group['mean'], yerr=group['ci'], fmt='none', color=p[list(data_avg[st.session_state.fac[1]].unique()).index(name)])

    plt.xlabel(st.session_state.fac[0])
    plt.ylabel(st.session_state.variable_dependante)
    plt.legend(title=st.session_state.fac[1], bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()

    # Afficher le plot dans Streamlit
    st.pyplot(plt)








