import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

st.markdown('# **Análisis A/B Web Vanguard** 📊')

st.sidebar.title('Opciones de Análisis')

grupo_seleccionado = st.sidebar.selectbox('Seleccione el grupo', ['Ambos', 'Test', 'Control'])

usuario_tipo = st.sidebar.radio('Seleccione el tipo de usuario', ['Todos', 'Lineal', 'No Lineal'])

paso_seleccionado = st.sidebar.selectbox('Seleccione el paso a analizar', ['Todos', 'start', 'step_1', 'step_2', 'step_3', 'confirm'])

@st.cache_data
def cargar_datos():
    df = pd.read_csv('data/cleaned/vanguard.csv', index_col=None)
    return df

df_vanguard = cargar_datos()

df_filtrado = df_vanguard.copy()

if grupo_seleccionado != 'Ambos':
    df_filtrado = df_filtrado[df_filtrado['variation'] == grupo_seleccionado]

if usuario_tipo == 'Lineal':
    df_filtrado = df_filtrado[df_filtrado['lineal'] == True]
elif usuario_tipo == 'No Lineal':
    df_filtrado = df_filtrado[df_filtrado['lineal'] == False]

if paso_seleccionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['step'] == paso_seleccionado]

def calculate_completion_rates(df):
    total_clientes_por_variacion = df.drop_duplicates(subset=['client_id', 'variation']).groupby('variation')['client_id'].nunique()

    lineal_true = df[(df['lineal'] == True)].drop_duplicates(subset=['client_id', 'variation'])
    lineal_false = df[(df['lineal'] == False) & (~df['client_id'].isin(lineal_true['client_id']))].drop_duplicates(subset=['client_id', 'variation'])

    completion_rate_lineal_true = lineal_true.groupby('variation')['client_id'].nunique() / total_clientes_por_variacion

    completion_rate_lineal_false = lineal_false.groupby('variation')['client_id'].nunique() / total_clientes_por_variacion

    suma_tasas = completion_rate_lineal_true + completion_rate_lineal_false

    return completion_rate_lineal_true, completion_rate_lineal_false, suma_tasas

def plot_completion_rates(completion_rate_true, completion_rate_false):
    completion_rates = pd.DataFrame({
        'Lineal=True': completion_rate_true,
        'Lineal=False': completion_rate_false
    })

    completion_rates.plot(kind='bar', figsize=(10, 6), color=['#4CAF50', '#FF5733'])

    plt.title('Tasa de Finalización y No Finalización por Grupo (Test vs Control)')
    plt.xlabel('Grupo')
    plt.ylabel('Proporción de Clientes')
    plt.xticks(rotation=0)
    plt.legend(title='Condición')
    plt.show()

def plot_avg_time_in_steps(df, step_order):
    df['lineal'] = df['lineal'].astype(bool)
    
    df['step'] = pd.Categorical(df['step'], categories=step_order, ordered=True)

    avg_time_in_step_lineal = df.groupby(['variation', 'lineal', 'step'])['total_time_in_step'].mean().unstack()

    avg_time_in_step_lineal = avg_time_in_step_lineal[step_order]

    plt.figure(figsize=(12, 6))

    for variation, lineal in avg_time_in_step_lineal.index:
        if lineal:
            label = f"{variation} - Lineal"
            marker = 'o'
            linestyle = '-'
        else:
            label = f"{variation} - No Lineal"
            marker = 'x'
            linestyle = '--'
        
        avg_time_in_step_lineal.loc[(variation, lineal), :].T.plot(
            kind='line', marker=marker, linestyle=linestyle, figsize=(12, 6), ax=plt.gca(), label=label
        )

    plt.title('Tiempo Promedio en Cada Paso: Test vs Control (Lineal vs No Lineal)')
    plt.xlabel('Paso')
    plt.ylabel('Tiempo Promedio (segundos)')
    plt.grid(True)
    plt.legend(title='Condición (Test/Control - Lineal/No Lineal)', loc='upper left')
    plt.show()

    return avg_time_in_step_lineal

def plot_error_rate(df, step_order):
    df['error'] = (df['step_diff'] == -1) | (df['step_repeat_count'] > 2)

    error_rate = df.groupby(['variation', 'lineal', 'step'])['error'].mean().unstack()

    error_rate = error_rate[step_order]

    return error_rate

def calculate_and_plot_time_stats(df, step_order):
    grouped_time_stats = df.groupby(['variation', 'step', 'lineal'])['total_time_in_step'].agg(
        mean='mean',
        skew=lambda x: skew(x),
        kurtosis=lambda x: kurtosis(x)
    )

    grouped_time_stats = grouped_time_stats.reindex(step_order, level='step')

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['mean'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Mean Time per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['skew'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Skewness per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Skewness')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['kurtosis'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Kurtosis per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Kurtosis')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    
    return grouped_time_stats

def calculate_and_plot_time_per_step(df, step_order):
    df_time_per_step = df.groupby(['variation', 'step'])['total_time_in_step'].mean().unstack()

    df_time_per_step = df_time_per_step[step_order]

    df_time_per_step.T.plot(kind='line', marker='o', figsize=(10, 6))
    plt.title('Tiempo Promedio por Paso: Test vs Control')
    plt.xlabel('Paso')
    plt.ylabel('Tiempo Promedio')
    plt.grid(True)
    plt.show()

    return df_time_per_step

def calculate_pearson_spearman_corr(df):
    pearson_corr = df[['balance', 'activity', 'age', 'num_accounts']].corr(method='pearson')
    spearman_corr = df[['balance', 'activity', 'age', 'num_accounts']].corr(method='spearman')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Heatmap')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Spearman Correlation Heatmap')
    plt.show()

    return pearson_corr, spearman_corr

def z_test_completion_rates(df):
    n_test = df[df['variation'] == 'Test']['client_id'].nunique()  # Número de usuarios en Test
    n_control = df[df['variation'] == 'Control']['client_id'].nunique()  # Número de usuarios en Control

    completed_test = df[(df['variation'] == 'Test') & (df['lineal'] == True)]['client_id'].nunique()  # Usuarios que completaron el proceso en Test
    completed_control = df[(df['variation'] == 'Control') & (df['lineal'] == True)]['client_id'].nunique()  # Usuarios que completaron el proceso en Control

    p_test = completed_test / n_test
    p_control = completed_control / n_control

    count = [completed_test, completed_control]
    nobs = [n_test, n_control]

    stat, p_value = proportions_ztest(count, nobs)

    return stat, p_value

def t_test_time_per_step(df, step):
    time_test_lineal = df[(df['variation'] == 'Test') & (df['lineal'] == True) & (df['step'] == step)]['total_time_in_step']
    time_control_lineal = df[(df['variation'] == 'Control') & (df['lineal'] == True) & (df['step'] == step)]['total_time_in_step']

    time_test_non_lineal = df[(df['variation'] == 'Test') & (df['lineal'] == False) & (df['step'] == step)]['total_time_in_step']
    time_control_non_lineal = df[(df['variation'] == 'Control') & (df['lineal'] == False) & (df['step'] == step)]['total_time_in_step']

    t_stat_lineal, p_value_lineal = stats.ttest_ind(time_test_lineal, time_control_lineal, equal_var=False)

    t_stat_non_lineal, p_value_non_lineal = stats.ttest_ind(time_test_non_lineal, time_control_non_lineal, equal_var=False)

    return (t_stat_lineal, p_value_lineal), (t_stat_non_lineal, p_value_non_lineal)

def mann_whitney_test_variation(df):
    test_group = df[df['variation'] == 'Test']['total_time_in_step']
    control_group = df[df['variation'] == 'Control']['total_time_in_step']

    u_statistic, p_value = stats.mannwhitneyu(test_group, control_group, alternative='two-sided')

    return u_statistic, p_value

def mann_whitney_test_lineal(df):
    lineal_group = df[df['lineal'] == True]['total_time_in_step']
    non_lineal_group = df[df['lineal'] == False]['total_time_in_step']

    u_statistic, p_value = stats.mannwhitneyu(lineal_group, non_lineal_group, alternative='two-sided')

    return u_statistic, p_value

def chi_square_and_cramers_v(df):
    contingency_table = pd.crosstab(df['variation'], df['lineal'])

    chi2, p, dof, ex = chi2_contingency(contingency_table)

    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    return chi2, p, cramers_v

df_vanguard = cargar_datos()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Introducción",
    "Tasa de Finalización", 
    "Tiempos por Paso", 
    "Tasa de Errores", 
    "Estadísticas de Tiempos", 
    "Correlaciones", 
    "Pruebas Estadísticas"
])

with tab1:
    st.markdown('''
    ## Introducción
    
    En este proyecto, llevamos a cabo un **análisis A/B** para evaluar el rendimiento comparativo entre la **nueva versión** de la web de Vanguard y su **versión anterior**. Este estudio nos permitirá medir el impacto de las mejoras introducidas en la nueva versión en la experiencia de usuario, utilizando métricas clave que nos ayuden a identificar si las modificaciones han resultado beneficiosas.
    ''')
                
    st.divider()    

    st.markdown('''
    El análisis se centra en tres métricas fundamentales:

    - **Tasa de finalización**: Analiza el porcentaje de usuarios que logran completar el proceso de navegación de principio a fin.
    - **Tiempos por paso**: Evalúa el tiempo promedio que los usuarios emplean en cada etapa del proceso, lo que nos proporciona información sobre posibles cuellos de botella o mejoras de eficiencia.
    - **Tasa de errores**: Mide la frecuencia con la que los usuarios cometen errores, como retroceder en pasos o repetir acciones, para detectar posibles dificultades en la interacción con la nueva interfaz.
    ''')
                
    st.divider()            

    st.markdown('''
    Para asegurar la rigurosidad del estudio, hemos empleado diversas **pruebas estadísticas** con el fin de determinar si las diferencias observadas entre ambas versiones son **estadísticamente significativas**:

    - **Pruebas T-test y Mann-Whitney**: Comparan las medias de tiempo de finalización entre los grupos de usuarios, evaluando si existen diferencias notables en el rendimiento.
    - **Pruebas Chi-square y Cramér's V**: Analizan la relación entre variables categóricas, como el tipo de usuario (lineal o no lineal) y la versión de la web utilizada (Test o Control).
    - **Correlaciones de Pearson y Spearman**: Examina la relación entre variables continuas como la **edad**, el **balance** de los usuarios, su **actividad** (número de logins y llamadas al soporte), y el **número de cuentas** que poseen, para detectar patrones de comportamiento que puedan influir en el rendimiento.
    ''')

    st.divider()            

    st.markdown('''
    El objetivo final de este análisis es determinar si la **nueva versión de la web** ofrece una **mejor experiencia de usuario**, basándonos en el comportamiento de los usuarios durante el proceso de navegación y en cómo estos interactúan con la interfaz. A través de este estudio, buscamos obtener insights valiosos que guíen futuras mejoras y optimizaciones en la experiencia digital de Vanguard.
    ''')

with tab2:
    st.subheader("Tasas de Finalización")
    
    st.write("""
    La tasa de finalización mide el porcentaje de usuarios que completaron el proceso de manera lineal. 
    El análisis se divide entre:
    
    - **Lineal=True**: Usuarios que completaron el proceso sin retroceder ni repetir pasos.
    - **Lineal=False**: Usuarios que retrocedieron o repitieron pasos más de dos veces.
    """)

    completion_rate_true, completion_rate_false, suma_tasas = calculate_completion_rates(df_filtrado)

    st.write("##### Tasa de finalización (lineal = True)")
    st.write(completion_rate_true)

    st.write("##### Tasa de no finalización (lineal = False)")
    st.write(completion_rate_false)

    st.write("##### Suma de tasas para cada grupo (debería ser cercana a 1)")
    st.write(suma_tasas)

    st.divider()

    st.subheader("Visualización")
    
    completion_rates = pd.DataFrame({
        'Lineal=True': completion_rate_true,
        'Lineal=False': completion_rate_false
    })

    completion_rates = completion_rates.reset_index()

    fig = px.bar(completion_rates, x='variation', y=['Lineal=True', 'Lineal=False'], barmode='group', title="Tasas de Finalización y No Finalización (Test vs Control - Lineal/No Lineal)")
    st.plotly_chart(fig)

    st.divider()

    st.write("""
    #### Insights
    - En el grupo **Control** (versión antigua de la web), el **46.05%** de los usuarios completaron el proceso de manera lineal, mientras que en el grupo **Test** (nueva versión), la tasa de finalización lineal aumentó ligeramente a **47.46%**. Aunque esta diferencia es pequeña, podría sugerir que la nueva versión facilita ligeramente un proceso más fluido para algunos usuarios.
    
    - Sin embargo, el porcentaje de usuarios que **no completaron** el proceso de manera lineal también se mantiene cercano entre los grupos. En el grupo **Control**, el **53.94%** de los usuarios no siguieron una ruta lineal, en comparación con el **52.53%** en el grupo **Test**.

    - Estos resultados sugieren que, aunque la nueva versión de la web presenta una leve mejora en la tasa de finalización lineal, el comportamiento no lineal sigue siendo común en ambas versiones. Esto podría indicar que existen desafíos inherentes en el flujo de navegación que no han sido completamente abordados con las modificaciones de la nueva versión.
    
    - Es importante analizar más en detalle qué factores contribuyen a que un porcentaje considerable de usuarios no complete el proceso de manera lineal. Algunas posibles áreas de mejora podrían incluir la claridad en los pasos del proceso, la accesibilidad de la interfaz, o el soporte proporcionado durante el proceso.
    """)

with tab3:
    st.subheader("Tiempos Promedio por Paso")
    
    st.write("""
    En este apartado se analiza el tiempo que los usuarios tardan en completar cada paso del proceso de navegación, 
    comparando los grupos Test (nueva versión de la web) y Control (versión antigua), y diferenciando entre usuarios lineales 
    (que completan el proceso sin errores) y no lineales (que retroceden o repiten pasos).
    """)

    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    avg_time_in_step_lineal = plot_avg_time_in_steps(df_filtrado, step_order)
    
    st.write(avg_time_in_step_lineal)

    st.divider()

    st.subheader("Visualización")

    df_grouped_time = df_filtrado.groupby(['variation', 'step', 'lineal'])['total_time_in_step'].mean().reset_index()

    df_grouped_time['step'] = pd.Categorical(df_grouped_time['step'], categories=step_order, ordered=True)

    fig_time = px.line(
        df_grouped_time, 
        x='step', 
        y='total_time_in_step', 
        color='variation', 
        line_dash='lineal',  # Agregar la variable lineal/no lineal como patrón de línea
        markers=True, 
        title="Tiempos Promedio por Paso (Test vs Control - Lineal/No Lineal)"
    )
    
    st.plotly_chart(fig_time)

    st.divider()

    st.write("""
    #### Insights
    - **Usuarios no lineales**: Los usuarios no lineales en el grupo **Test** tardan más en completar los primeros pasos, lo que podría indicar problemas de usabilidad o confusión inicial con la nueva versión de la web. Este retraso sugiere la necesidad de una revisión en las primeras etapas del flujo para identificar puntos críticos que dificultan el avance de estos usuarios.
    
    - **Usuarios lineales**: Los usuarios lineales del grupo **Test** completan el proceso más rápido que los del grupo **Control**, lo que refuerza la idea de que, para aquellos que navegan sin errores, la nueva versión de la web ofrece una experiencia más eficiente y optimizada.
    
    - **Pasos intermedios**: Los tiempos en los pasos **Step 2** y **Step 3** son significativamente más largos para ambos grupos, lo que refleja una mayor complejidad en estas etapas. Este hallazgo sugiere que estos pasos podrían ser un área de mejora clave para optimizar el flujo general, ya que su complejidad está afectando tanto a los usuarios lineales como a los no lineales.
    
    - **Paso Confirm**: La nueva versión de la web reduce los tiempos en el paso final (**Confirm**) para los usuarios no lineales, lo que podría indicar que, a medida que los usuarios avanzan en el proceso, se adaptan mejor a la nueva interfaz, o que las mejoras en este paso en particular son más efectivas para este grupo.
    
    - **Conclusión general**: Aunque la nueva versión de la web parece mejorar la eficiencia general para los usuarios lineales, los usuarios no lineales aún enfrentan desafíos importantes en las primeras etapas del proceso. Mejorar la accesibilidad y claridad en los pasos iniciales, especialmente para los no lineales, podría aumentar la tasa de finalización y reducir los tiempos totales.
    """)

with tab4:
    st.subheader("Tasas de Error")
    
    st.write("""
    La tasa de errores mide cuántos usuarios experimentaron dificultades durante el proceso, ya sea por retroceder 
    en los pasos o por repetirlos más de dos veces. Este análisis es esencial para identificar los puntos críticos 
    donde los usuarios tienen problemas, tanto en la versión antigua de la web (grupo Control) como en la nueva (grupo Test).
    """)

    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    error_rate = plot_error_rate(df_filtrado, step_order)
    
    error_rate_reset = error_rate.reset_index()

    error_rate_melted = error_rate_reset.melt(id_vars=['variation', 'lineal'], value_vars=step_order, var_name='step', value_name='error_rate')

    error_rate_filtered = error_rate_melted[error_rate_melted['lineal'] == False]

    st.write(error_rate_filtered)

    st.divider()

    st.subheader("Visualización")
    fig_error = px.line(
        error_rate_melted, 
        x='step', 
        y='error_rate', 
        color='variation', 
        line_dash='lineal',  # Diferenciar lineales y no lineales con líneas diferentes
        markers=True,
        title="Tasa de Errores por Grupo (Test vs Control - Lineal/No Lineal)",
        labels={"error_rate": "Tasa de Error"}
    )
    st.plotly_chart(fig_error)

    st.divider()

    st.write("""
    #### Insights
    - **Paso Start y usuarios no lineales**: La tasa de errores es considerablemente más alta en el paso **Start** para los usuarios **no lineales** en el grupo **Test** (34.62%) en comparación con el grupo **Control** (23.37%). Esto indica que la nueva versión de la web podría estar presentando barreras significativas para los usuarios al inicio del proceso, lo que podría estar afectando la capacidad de estos usuarios para avanzar de manera efectiva.
    
    - **Evolución de las tasas de error**: En los pasos siguientes, la tasa de errores disminuye tanto en el grupo **Test** como en el grupo **Control**. En el grupo **Test**, la tasa baja al **20.85%** en **Step 1** y continúa disminuyendo en **Step 2** (**21.29%**) y **Step 3** (**7.91%**). En el grupo **Control**, el patrón es similar, pero las tasas de error se mantienen más consistentes entre **Step 1** (**15.18%**) y **Step 2** (**22.89%**), lo que sugiere que, aunque la nueva versión tiene problemas al inicio, se comporta mejor en los últimos pasos.
    
    - **Paso Confirm**: Curiosamente, la tasa de errores en el paso final **Confirm** es ligeramente superior en el grupo **Test** (**8.78%**) en comparación con el grupo **Control** (**7.68%**), lo que podría señalar alguna confusión residual en la nueva versión al completar el proceso.
    
    - **Usuarios lineales**: No se detectaron errores en ningún paso para los usuarios **lineales** en ambos grupos, lo que confirma que, para aquellos que navegan de manera fluida y sin repeticiones, la experiencia es completamente libre de problemas en ambos casos. Esto sugiere que las dificultades están más concentradas en los usuarios que retroceden o repiten pasos, probablemente debido a falta de claridad o problemas de usabilidad.
    
    - **Conclusión general**: La **nueva versión de la web** presenta más dificultades en el paso **Start** para los usuarios no lineales, lo que podría estar causando una experiencia inicial negativa que impacta en el progreso posterior. Sin embargo, en los pasos intermedios y finales, las tasas de error disminuyen en el grupo **Test**, lo que sugiere que la nueva versión se vuelve más intuitiva a medida que los usuarios avanzan en el proceso. Se recomienda mejorar la experiencia inicial en el paso **Start** para minimizar los errores y evitar un impacto negativo en la percepción general.
    """)

with tab5:
    st.subheader("Estadísticas de Tiempos: Media, Skewness y Kurtosis")
    
    st.write("""
    Además del análisis de los tiempos promedio, es crucial analizar la forma de la distribución de los tiempos 
    en cada paso para obtener una visión más detallada del comportamiento de los usuarios. 
    Para ello, se calculan las siguientes métricas:
    
    - **Media**: El tiempo promedio que los usuarios dedican a cada paso.
    - **Skewness (asimetría)**: Indica si los tiempos están sesgados hacia un lado de la media.
    - **Kurtosis (curtosis)**: Mide la "cola" de la distribución, mostrando si hay valores extremos de tiempo.
    """)

    st.divider()

    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    grouped_time_stats = calculate_and_plot_time_stats(df_filtrado, step_order)

    grouped_time_stats = grouped_time_stats.reset_index()

    st.write(grouped_time_stats)

    st.divider()

    st.subheader("Visualización")

    fig_mean = px.bar(grouped_time_stats, x='step', y='mean', color='variation', 
                      barmode='group', title='Media de Tiempos por Grupo y Paso',
                      labels={'mean': 'Tiempo promedio (s)', 'step': 'Paso'})
    st.plotly_chart(fig_mean)

    fig_skewness = px.bar(grouped_time_stats, x='step', y='skew', color='variation', 
                          barmode='group', title='Skewness por Grupo y Paso',
                          labels={'skew': 'Skewness', 'step': 'Paso'})
    st.plotly_chart(fig_skewness)

    fig_kurtosis = px.bar(grouped_time_stats, x='step', y='kurtosis', color='variation', 
                          barmode='group', title='Kurtosis por Grupo y Paso',
                          labels={'kurtosis': 'Kurtosis', 'step': 'Paso'})
    st.plotly_chart(fig_kurtosis)

    st.divider()

    st.markdown("""
    ### Insights

    1. **Media (mean)**:
   - En general, los **usuarios no lineales** tienden a pasar **más tiempo** en todos los pasos del proceso, especialmente en los pasos intermedios y finales. Esto es particularmente evidente en el grupo **Test**, donde los usuarios no lineales pasan, por ejemplo, 194 segundos en el paso **step_3**, frente a 157 segundos para los lineales.
   - Los **usuarios lineales del grupo Control** completan el proceso de manera más rápida en los primeros pasos (**start** y **step_1**), con una media de 12 y 37 segundos respectivamente, comparado con el grupo **Test** donde los tiempos son de 7 y 30 segundos en los mismos pasos. Esto sugiere que el grupo Test tiene un inicio algo más rápido, pero se ralentiza en los pasos posteriores.
   - En el **paso confirm**, los usuarios lineales del grupo **Control** tienen un tiempo promedio superior (290 segundos) en comparación con el grupo **Test** (250 segundos), lo que sugiere que la nueva versión mejora la eficiencia en la fase final del proceso.

    2. **Skewness (asimetría)**:
   - En ambos grupos, la **skewness** es extremadamente alta en los primeros pasos (**start** y **step_1**), especialmente para los **usuarios no lineales**, lo que sugiere que hay una **minoría significativa de usuarios** que tarda mucho más que el promedio en completar estos pasos. Esto es más notable en el grupo **Control** en el paso **start**, con un skew de 53 para usuarios no lineales.
   - La asimetría disminuye a medida que los usuarios avanzan en los pasos. Sin embargo, persiste en el paso **confirm** para los usuarios no lineales, particularmente en el grupo **Test**, donde la asimetría es alta (9.27), lo que indica que aún hay usuarios que experimentan dificultades significativas en los pasos finales de la nueva versión de la web.

    3. **Kurtosis**:
   - Los valores extremadamente altos de **kurtosis** en los primeros pasos, especialmente para los **usuarios no lineales**, indican la presencia de **valores atípicos extremos**. Por ejemplo, el paso **start** en el grupo **Control** tiene una kurtosis de 5255 para los usuarios no lineales, lo que sugiere que una pequeña proporción de usuarios está experimentando tiempos muy prolongados en este paso.
   - En el grupo **Test**, la **kurtosis** es más alta en los pasos **step_1** y **confirm**, con valores de 403 y 280 respectivamente, lo que indica que algunos usuarios enfrentan **dificultades graves** al completar estos pasos en la nueva versión.
   - Los **usuarios lineales** en ambos grupos muestran una **kurtosis más baja** en comparación con los no lineales, lo que confirma que tienen una experiencia más consistente y sin grandes retrasos.
    """)

with tab6:
    st.subheader("Correlaciones entre Variables")
    
    st.write("""
    En esta sección se analizan las correlaciones entre diferentes variables relacionadas con los clientes, 
    como **balance**, **edad**, **actividad** (suma de logins y llamadas a soporte), y **número de cuentas**. 
    El objetivo es descubrir relaciones significativas que puedan influir en el comportamiento de los usuarios 
    y su rendimiento en el proceso de navegación en la web.
    """)

    st.divider()
    
    st.write("""
    ### Correlaciones de Pearson y Spearman
    Se utilizan dos métodos para calcular las correlaciones:
    
    - **Correlación de Pearson**: Mide la relación lineal entre dos variables continuas. 
      Un valor cercano a 1 indica una fuerte correlación positiva, mientras que un valor cercano a -1 indica 
      una fuerte correlación negativa.
    - **Correlación de Spearman**: Mide la correlación monótona entre dos variables, 
      lo que permite detectar relaciones que no son estrictamente lineales.
    """)

    st.divider()

    pearson_corr, spearman_corr = calculate_pearson_spearman_corr(df_filtrado)

    st.subheader("Matriz de Correlación de Pearson")
    st.write(pearson_corr)

    st.subheader("Matriz de Correlación de Spearman")
    st.write(spearman_corr)

    st.divider()

    st.subheader("Visualización")

    fig_pearson = px.imshow(pearson_corr, text_auto=True, aspect="auto", title="Correlación de Pearson")
    st.plotly_chart(fig_pearson)

    fig_spearman = px.imshow(spearman_corr, text_auto=True, aspect="auto", title="Correlación de Spearman")
    st.plotly_chart(fig_spearman)

    st.divider()

    st.markdown("""
    ### Insights

    1. **Balance y Número de Cuentas**:
   - La correlación entre **balance** y **número de cuentas** es moderadamente positiva tanto en Pearson (**0.26**) como en Spearman (**0.33**), lo que sugiere que los clientes con más cuentas tienden a tener balances más altos. La correlación más fuerte en Spearman indica que la relación podría no ser estrictamente lineal, pero sigue siendo significativa.

    2. **Balance y Edad**:
   - Existe una correlación positiva entre **balance** y **edad**, especialmente en Spearman (**0.35**), lo que sugiere que los clientes mayores tienden a tener balances más altos. La mayor correlación en Spearman refleja que la relación entre estas variables es más sólida cuando se consideran rangos de edad en lugar de un cambio lineal estricto.

    3. **Balance y Actividad**:
   - La correlación entre **balance** y **actividad** es positiva pero baja, tanto en Pearson (**0.17**) como en Spearman (**0.29**). Esto indica que los clientes con balances más altos tienden a ser ligeramente más activos, pero la actividad no es un factor fuerte para predecir el balance.

    4. **Actividad y Número de Cuentas**:
   - Existe una correlación moderada entre **actividad** y **número de cuentas**, con valores de Pearson (**0.24**) y Spearman (**0.24**). Esto sugiere que los clientes con más cuentas tienden a ser más activos, aunque no de manera contundente.

    5. **Edad y Actividad**:
   - La correlación entre **edad** y **actividad** es muy baja tanto en Pearson (**0.08**) como en Spearman (**0.08**), lo que indica que la **edad** no tiene un impacto significativo en el nivel de actividad de los clientes. Los clientes de todas las edades tienden a mostrar patrones similares en cuanto a su actividad en la web.
    """)

with tab7:
    st.subheader("Pruebas Estadísticas")

    st.write("""
    Para validar los resultados obtenidos y confirmar si las diferencias observadas entre los grupos 
    **Test** (nueva versión de la web) y **Control** (versión antigua) son estadísticamente significativas, 
    realizamos una serie de pruebas estadísticas. Estas pruebas nos permiten determinar si las diferencias 
    observadas en las métricas clave (como tasas de finalización y tiempos por paso) son atribuibles a la nueva versión 
    de la web o si son producto del azar.
    """)

    st.divider()

    st.write("""
    ### Tipos de Pruebas Realizadas:
    
    - **Prueba Z**: Comparación de las tasas de finalización entre los grupos Test y Control.
    - **T-test**: Comparación de los tiempos de finalización entre los usuarios de Test y Control, tanto para usuarios lineales como no lineales.
    - **Prueba Mann-Whitney**: Evaluación de diferencias en los tiempos de finalización entre los grupos cuando los datos no siguen una distribución normal.
    - **Prueba Chi-square**: Evaluación de la asociación entre las variables categóricas, como Test/Control y Lineal/No Lineal.
    - **Cramér's V**: Medición de la fuerza de la asociación entre variables categóricas.
    """)

    st.divider()

    st.markdown("#### 1. Prueba Z para Tasas de Finalización")
    stat, p_value = z_test_completion_rates(df_vanguard)
    st.write(f"**Estadístico Z**: {stat:.3f}")
    st.write(f"**Valor p**: {p_value:.5f}")
    
    if p_value < 0.05:
        st.write("La diferencia en las tasas de finalización entre Test y Control es **estadísticamente significativa**.")
    else:
        st.write("La diferencia en las tasas de finalización **no es estadísticamente significativa**.")

    st.markdown("""
    ##### Insight
    - El estadístico Z muestra una diferencia significativa en las tasas de finalización, con un valor p de **0.0015**, lo que indica que la nueva versión de la web ha afectado significativamente el comportamiento de finalización de los usuarios.
    """)

    st.divider()

    st.markdown("#### 2. T-test para Tiempos de Finalización")
    step = 'confirm'
    (t_stat_lineal, p_value_lineal), (t_stat_non_lineal, p_value_non_lineal) = t_test_time_per_step(df_vanguard, step)

    st.write("**Resultados para usuarios lineales:**")
    st.write(f"**Estadístico t**: {t_stat_lineal:.2f}")
    st.write(f"**Valor p**: {p_value_lineal:.2e}")
    if p_value_lineal < 0.05:
        st.write("La diferencia en los tiempos de finalización para usuarios lineales es **estadísticamente significativa**.")
    else:
        st.write("La diferencia en los tiempos de finalización para usuarios lineales **no es estadísticamente significativa**.")
    
    st.write("**Resultados para usuarios no lineales:**")
    st.write(f"**Estadístico t**: {t_stat_non_lineal:.2f}")
    st.write(f"**Valor p**: {p_value_non_lineal:.2e}")
    if p_value_non_lineal < 0.05:
        st.write("La diferencia en los tiempos de finalización para usuarios no lineales es **estadísticamente significativa**.")
    else:
        st.write("La diferencia en los tiempos de finalización para usuarios no lineales **no es estadísticamente significativa**.")

    st.markdown("""
    ##### Insight
    - Para los usuarios **lineales**, el valor p es extremadamente bajo (**2.18e-40**), lo que indica una diferencia estadísticamente significativa entre Test y Control.
    - Para los usuarios **no lineales**, el valor p también es muy bajo (**8.11e-21**), lo que sugiere que la nueva versión de la web impacta notablemente el comportamiento de estos usuarios.
    """)

    st.divider()

    st.markdown("#### 3. Prueba Mann-Whitney U para Tiempos de Finalización")
    u_statistic_variation, p_value_variation = mann_whitney_test_variation(df_vanguard)
    st.write(f"**U-statistic (Test vs Control)**: {u_statistic_variation:,.0f}")
    st.write(f"**P-value (Test vs Control)**: {p_value_variation:.2e}")
    
    if p_value_variation < 0.05:
        st.write("La diferencia entre los grupos Test y Control es **estadísticamente significativa**.")
    else:
        st.write("No hay una diferencia estadísticamente significativa entre los grupos Test y Control.")
    
    u_statistic_lineal, p_value_lineal = mann_whitney_test_lineal(df_vanguard)
    st.write(f"**U-statistic (Lineal vs No Lineal)**: {u_statistic_lineal:,.0f}")
    st.write(f"**P-value (Lineal vs No Lineal)**: {p_value_lineal:.2e}")
    
    if p_value_lineal < 0.05:
        st.write("La diferencia entre usuarios lineales y no lineales es **estadísticamente significativa**.")
    else:
        st.write("No hay una diferencia estadísticamente significativa entre usuarios lineales y no lineales.")

    st.markdown("""
    ##### Insight
    - La prueba Mann-Whitney U muestra que la diferencia en los tiempos de finalización entre Test y Control es **estadísticamente significativa** (valor p de **1.59e-08**).
    - La diferencia entre usuarios **lineales** y **no lineales** es altamente significativa, con un valor p de **7.29e-277**, lo que destaca una marcada diferencia en su comportamiento.
    """)

    st.divider()

    st.markdown("#### 4. Prueba Chi-square y Cramér's V")
    chi2, p_value, cramers_v = chi_square_and_cramers_v(df_vanguard)
    st.write(f"**Chi-square statistic**: {chi2:.2f}")
    st.write(f"**P-value**: {p_value:.2e}")
    st.write(f"**Cramér's V**: {cramers_v:.4f}")
    
    if p_value < 0.05:
        st.write("La asociación entre 'variation' y 'lineal' es **estadísticamente significativa**.")
    else:
        st.write("No hay una asociación estadísticamente significativa entre 'variation' y 'lineal'.")

    st.markdown("""
    ##### Insight
    - El estadístico Chi-square es **133.73** con un valor p de **6.26e-31**, lo que indica que hay una asociación significativa entre los grupos Test/Control y la categorización lineal/no lineal.
    - El valor de **Cramér's V** es **0.0205**, lo que indica que aunque la asociación es estadísticamente significativa, la fuerza de la relación entre las variables es débil.
    """)