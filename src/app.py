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

# Título con Markdown
st.markdown('# **Análisis A/B Web Vanguard** 📈')

# Sidebar para opciones
st.sidebar.title('Opciones del Análisis')

# Filtrar por grupo (Test/Control)
grupo_seleccionado = st.sidebar.selectbox('Seleccione el grupo', ['Ambos', 'Test', 'Control'])

# Filtrar por tipo de usuario (Lineal/No Lineal)
usuario_tipo = st.sidebar.radio('Seleccione el tipo de usuario', ['Todos', 'Lineal', 'No Lineal'])

# Selección de paso específico
paso_seleccionado = st.sidebar.selectbox('Seleccione el paso a analizar', ['Todos', 'start', 'step_1', 'step_2', 'step_3', 'confirm'])

# Función para cargar los datos con cache
@st.cache_data
def cargar_datos():
    df = pd.read_csv('../data/cleaned/vanguard.csv', index_col=None)  # Evita cargar un índice extra
    return df

# Cargar los datos
df_vanguard = cargar_datos()

# Crear una copia del DataFrame original para aplicar los filtros
df_filtrado = df_vanguard.copy()

# Filtrar los datos según el grupo seleccionado
if grupo_seleccionado != 'Ambos':
    df_filtrado = df_filtrado[df_filtrado['variation'] == grupo_seleccionado]

# Filtrar los datos según el tipo de usuario seleccionado
if usuario_tipo == 'Lineal':
    df_filtrado = df_filtrado[df_filtrado['lineal'] == True]
elif usuario_tipo == 'No Lineal':
    df_filtrado = df_filtrado[df_filtrado['lineal'] == False]

# Filtrar los datos según el paso seleccionado
if paso_seleccionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['step'] == paso_seleccionado]

def calculate_completion_rates(df):
    total_clientes_por_variacion = df.drop_duplicates(subset=['client_id', 'variation']).groupby('variation')['client_id'].nunique()

    # Asegurarse de que no hay solapamiento entre clientes "lineal=True" y "lineal=False"
    lineal_true = df[(df['lineal'] == True)].drop_duplicates(subset=['client_id', 'variation'])
    lineal_false = df[(df['lineal'] == False) & (~df['client_id'].isin(lineal_true['client_id']))].drop_duplicates(subset=['client_id', 'variation'])

    # Calculamos la tasa de finalización para los que completaron el proceso de manera lineal
    completion_rate_lineal_true = lineal_true.groupby('variation')['client_id'].nunique() / total_clientes_por_variacion

    # Calculamos la tasa de "no finalización" para los que no completaron el proceso de manera lineal
    completion_rate_lineal_false = lineal_false.groupby('variation')['client_id'].nunique() / total_clientes_por_variacion

    # Suma de tasas
    suma_tasas = completion_rate_lineal_true + completion_rate_lineal_false

    return completion_rate_lineal_true, completion_rate_lineal_false, suma_tasas

def plot_completion_rates(completion_rate_true, completion_rate_false):
    completion_rates = pd.DataFrame({
        'Lineal=True': completion_rate_true,
        'Lineal=False': completion_rate_false
    })

    # Generar el gráfico de barras
    completion_rates.plot(kind='bar', figsize=(10, 6), color=['#4CAF50', '#FF5733'])

    # Añadir etiquetas y título
    plt.title('Tasa de Finalización y No Finalización por Grupo (Test vs Control)')
    plt.xlabel('Grupo')
    plt.ylabel('Proporción de Clientes')
    plt.xticks(rotation=0)
    plt.legend(title='Condición')
    plt.show()

def plot_avg_time_in_steps(df, step_order):
    # Asegurarse de que la columna 'lineal' es booleana
    df['lineal'] = df['lineal'].astype(bool)
    
    # Asegurarse de que los pasos están correctamente categorizados
    df['step'] = pd.Categorical(df['step'], categories=step_order, ordered=True)

    # Agrupar por 'variation', 'lineal', y 'step' para calcular el tiempo promedio en cada paso
    avg_time_in_step_lineal = df.groupby(['variation', 'lineal', 'step'])['total_time_in_step'].mean().unstack()

    # Reindexar los pasos para reflejar el orden correcto
    avg_time_in_step_lineal = avg_time_in_step_lineal[step_order]

    # Crear un gráfico que muestre las cuatro combinaciones
    plt.figure(figsize=(12, 6))

    # Graficar los datos para todas las combinaciones (Test/Control y Lineal/No Lineal)
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

    # Añadir etiquetas y título
    plt.title('Tiempo Promedio en Cada Paso: Test vs Control (Lineal vs No Lineal)')
    plt.xlabel('Paso')
    plt.ylabel('Tiempo Promedio (segundos)')
    plt.grid(True)
    plt.legend(title='Condición (Test/Control - Lineal/No Lineal)', loc='upper left')
    plt.show()

    return avg_time_in_step_lineal

def plot_error_rate(df, step_order):
    # Definir una nueva columna para marcar los errores (retrocesos o repeticiones mayores a 2)
    df['error'] = (df['step_diff'] == -1) | (df['step_repeat_count'] > 2)

    # Agrupar por 'variation', 'lineal', y 'step' para calcular la tasa de errores
    error_rate = df.groupby(['variation', 'lineal', 'step'])['error'].mean().unstack()

    # Reordenar las columnas para reflejar el orden correcto de los pasos
    error_rate = error_rate[step_order]

    return error_rate

def calculate_and_plot_time_stats(df, step_order):
    grouped_time_stats = df.groupby(['variation', 'step', 'lineal'])['total_time_in_step'].agg(
        mean='mean',
        skew=lambda x: skew(x),
        kurtosis=lambda x: kurtosis(x)
    )

    # Reindexar el nivel de 'step' en el orden deseado
    grouped_time_stats = grouped_time_stats.reindex(step_order, level='step')

    # Graficar la media (mean)
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['mean'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Mean Time per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Graficar el skewness
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['skew'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Skewness per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Skewness')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Graficar la kurtosis
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

    # Reordenar los pasos
    df_time_per_step = df_time_per_step[step_order]

    # Graficar los tiempos promedio por paso
    df_time_per_step.T.plot(kind='line', marker='o', figsize=(10, 6))
    plt.title('Tiempo Promedio por Paso: Test vs Control')
    plt.xlabel('Paso')
    plt.ylabel('Tiempo Promedio')
    plt.grid(True)
    plt.show()

    # Retornar el DataFrame con los tiempos promedio
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

# Cargar los datos
df_vanguard = cargar_datos()

# Tabs para organizar la app
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
    ## Introducción al Proyecto de Análisis A/B
    
    En este proyecto, realizamos un análisis A/B para comparar la **nueva versión** de la web de Vanguard con la **versión antigua**.
    El objetivo es evaluar el rendimiento de la nueva web a través de varias métricas clave como:
    
    - **Tasa de finalización**: Proporción de usuarios que completan el proceso.
    - **Tiempos por paso**: Tiempo promedio que los usuarios tardan en completar cada paso del proceso.
    - **Tasa de errores**: Frecuencia de errores cometidos por los usuarios.
    
    También utilizamos diversas **pruebas estadísticas** para determinar si las diferencias observadas son **estadísticamente significativas**:
    
    - **T-test** y **Mann-Whitney**: Comparación de medias de tiempo entre grupos.
    - **Chi-square** y **Cramér's V**: Análisis de asociación entre las variables categóricas.
    - **Correlaciones de Pearson y Spearman**: Evaluamos la relación entre variables continuas como la actividad, edad, número de cuentas y balances de los clientes.
    
    Los resultados de este análisis nos permiten determinar si la nueva versión de la web ofrece una **mejor experiencia** para los usuarios, basándonos en su comportamiento durante el proceso de navegación.
    ''')

with tab2:
    st.subheader("Tasa de Finalización")
    
    st.write("""
    La tasa de finalización mide el porcentaje de usuarios que completaron el proceso de manera lineal. 
    El análisis se divide entre:
    
    - **Lineal=True**: Usuarios que completaron el proceso sin retroceder ni repetir pasos.
    - **Lineal=False**: Usuarios que retrocedieron o repitieron pasos más de dos veces.
    """)

    # Llamar a la función para calcular las tasas
    completion_rate_true, completion_rate_false, suma_tasas = calculate_completion_rates(df_filtrado)

    # Mostrar las tasas calculadas
    st.write("#### Tasa de finalización (lineal=True)")
    st.write(completion_rate_true)

    st.write("#### Tasa de no finalización (lineal=False)")
    st.write(completion_rate_false)

    st.write("#### Suma de tasas para cada grupo (debería ser cercana a 1)")
    st.write(suma_tasas)

    # Mostrar insights sobre las tasas de finalización
    st.write("""
    #### Insights:
    - En el grupo **Control** (versión antigua de la web), el **46.05%** de los usuarios completaron el proceso de manera lineal.
    - En el grupo **Test** (nueva versión de la web), el **47.46%** de los usuarios completaron el proceso de manera lineal.
    - En el grupo **Control**, el **53.94%** de los usuarios no completaron el proceso de manera lineal.
    - En el grupo **Test**, el **52.53%** de los usuarios no completaron el proceso de manera lineal.
    """)

    # Crear un gráfico de barras para visualizar las tasas de finalización
    st.subheader("Visualización de las Tasas de Finalización y No Finalización")
    st.write("Gráfico comparativo de las tasas de finalización por grupo (Test vs Control):")
    
    # Crear un dataframe para visualización
    completion_rates = pd.DataFrame({
        'Lineal=True': completion_rate_true,
        'Lineal=False': completion_rate_false
    })

    # Ajustar el índice del DataFrame para que sea 'variation' (Test vs Control)
    completion_rates = completion_rates.reset_index()

    # Generar gráfico de barras con Plotly
    fig = px.bar(completion_rates, x='variation', y=['Lineal=True', 'Lineal=False'], barmode='group', title="Tasas de Finalización y No Finalización (Test vs Control)")
    st.plotly_chart(fig)

with tab3:
    st.subheader("4.1 Tiempos Promedio por Paso")
    
    # Introducción al análisis de tiempos
    st.write("""
    En este apartado se analiza el tiempo que los usuarios tardan en completar cada paso del proceso de navegación, 
    comparando los grupos Test (nueva versión de la web) y Control (versión antigua), y diferenciando entre usuarios lineales 
    (que completan el proceso sin errores) y no lineales (que retroceden o repiten pasos).
    """)

    # Definir el orden correcto de los pasos
    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    # Llamar a la función para tiempos promedio
    avg_time_in_step_lineal = plot_avg_time_in_steps(df_filtrado, step_order)
    
    # Mostrar los datos procesados
    st.write("Tiempos promedio por paso y grupo:")
    st.write(avg_time_in_step_lineal)

    # Crear un gráfico interactivo de líneas para visualizar los tiempos promedio por paso
    st.subheader("Visualización de Tiempos Promedio por Paso")

    # Organizar los datos para el gráfico
    df_grouped_time = df_filtrado.groupby(['variation', 'step', 'lineal'])['total_time_in_step'].mean().reset_index()

    # Asegurarse de que los pasos siguen el orden correcto
    df_grouped_time['step'] = pd.Categorical(df_grouped_time['step'], categories=step_order, ordered=True)

    # Crear el gráfico con Plotly para visualizar las cuatro líneas
    fig_time = px.line(
        df_grouped_time, 
        x='step', 
        y='total_time_in_step', 
        color='variation', 
        line_dash='lineal',  # Agregar la variable lineal/no lineal como patrón de línea
        markers=True, 
        title="Tiempos Promedio por Paso (Test vs Control - Lineal/No Lineal)"
    )
    
    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig_time)


    # Mostrar Insights del análisis de tiempos
    st.write("""
    #### Insights:
    - **Usuarios no lineales**: Los usuarios no lineales en el grupo **Test** tardan más tiempo en completar los primeros pasos, 
      lo que sugiere posibles dificultades en la nueva versión.
    - **Usuarios lineales**: Los usuarios lineales del grupo **Test** completan el proceso de manera más eficiente que los de 
      **Control**, lo que sugiere que la nueva versión de la web es más eficiente para aquellos que no cometen errores.
    - **Pasos intermedios**: Los pasos **Step 2** y **Step 3** son más largos en ambos grupos, lo que refleja la complejidad de estas 
      etapas en el proceso.
    - **Paso Confirm**: La nueva versión de la web ayuda a reducir el tiempo en el paso final (**Confirm**) para los usuarios no lineales.
    """)

with tab4:
    st.subheader("3.2 Tasa de Errores")
    
    # Introducción al análisis de errores
    st.write("""
    La tasa de errores mide cuántos usuarios experimentaron dificultades durante el proceso, ya sea por retroceder 
    en los pasos o por repetirlos más de dos veces. Este análisis es esencial para identificar los puntos críticos 
    donde los usuarios tienen problemas, tanto en la versión antigua de la web (grupo Control) como en la nueva (grupo Test).
    """)

    # Definir el orden correcto de los pasos
    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    # Llamar a la función para calcular la tasa de errores
    error_rate = plot_error_rate(df_filtrado, step_order)
    
    # Resetear el índice para asegurarnos de tener todas las columnas necesarias
    error_rate_reset = error_rate.reset_index()

    # Hacer melt del DataFrame para organizar los datos correctamente
    error_rate_melted = error_rate_reset.melt(id_vars=['variation', 'lineal'], value_vars=step_order, var_name='step', value_name='error_rate')

    # Visualización del gráfico de la tasa de errores
    st.subheader("Visualización de la Tasa de Errores por Grupo y Tipo de Usuario")
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

    # Insights sobre la tasa de errores
    st.write("""
    #### Insight:
    - La tasa de errores es más alta en el paso **Start** para los usuarios **no lineales** en el grupo **Test** (34.62%) que en el grupo **Control** (23.37%). 
    Esto podría indicar problemas en la nueva versión de la web al inicio del proceso.
    
    - Los usuarios **Test** muestran tasas de error más altas en los primeros pasos, pero tasas más bajas en los últimos pasos (**Step 3** y **Confirm**) 
    comparado con **Control**. Esto sugiere una confusión inicial, pero mejora en las etapas finales.
    
    - **Usuarios lineales**: No se detectaron errores en ninguno de los pasos para los usuarios lineales en ambos grupos, 
    lo que indica que aquellos que siguen el proceso correctamente no encuentran problemas significativos.
    """)

with tab5:
    st.subheader("4. Estadísticas de Tiempos: Media, Skewness, y Kurtosis")
    
    st.write("""
    Además del análisis de los tiempos promedio, es crucial analizar la forma de la distribución de los tiempos 
    en cada paso para obtener una visión más detallada del comportamiento de los usuarios. 
    Para ello, se calculan las siguientes métricas:
    
    - **Media**: El tiempo promedio que los usuarios dedican a cada paso.
    - **Skewness (asimetría)**: Indica si los tiempos están sesgados hacia un lado de la media.
    - **Kurtosis (curtosis)**: Mide la "cola" de la distribución, mostrando si hay valores extremos de tiempo.
    """)

    # Definir el orden correcto de los pasos
    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    # Llamar a la función para estadísticas de tiempos y mostrar los datos
    grouped_time_stats = calculate_and_plot_time_stats(df_filtrado, step_order)

    # Resetear el índice (para evitar MultiIndex)
    grouped_time_stats = grouped_time_stats.reset_index()

    # Mostrar el DataFrame de estadísticas
    st.write("#### Estadísticas de tiempos (Media, Skewness, y Kurtosis)")
    st.write(grouped_time_stats)

    # Graficar las métricas (Media, Skewness y Kurtosis) con Plotly
    st.subheader("Visualización de Estadísticas de Tiempos por Grupo y Paso")

    # Generar gráfico de barras para la media
    fig_mean = px.bar(grouped_time_stats, x='step', y='mean', color='variation', 
                      barmode='group', title='Media de Tiempos por Grupo y Paso',
                      labels={'mean': 'Tiempo promedio (s)', 'step': 'Paso'})
    st.plotly_chart(fig_mean)

    # Generar gráfico de barras para la skewness
    fig_skewness = px.bar(grouped_time_stats, x='step', y='skew', color='variation', 
                          barmode='group', title='Skewness por Grupo y Paso',
                          labels={'skew': 'Skewness', 'step': 'Paso'})
    st.plotly_chart(fig_skewness)

    # Generar gráfico de barras para la kurtosis
    fig_kurtosis = px.bar(grouped_time_stats, x='step', y='kurtosis', color='variation', 
                          barmode='group', title='Kurtosis por Grupo y Paso',
                          labels={'kurtosis': 'Kurtosis', 'step': 'Paso'})
    st.plotly_chart(fig_kurtosis)

    # Mostrar insights
    st.write("""
    #### Insights:
    1. **Media (mean)**: Los usuarios no lineales del grupo Test tienden a pasar más tiempo en los pasos iniciales, 
       mientras que los usuarios lineales completan los pasos más rápido en el grupo Test que en el grupo Control.
       
    2. **Skewness**: La asimetría es alta en todos los pasos, especialmente en los pasos iniciales, lo que sugiere 
       que algunos usuarios tardan mucho más en completarlos.
       
    3. **Kurtosis**: La alta kurtosis en ambos grupos indica la presencia de valores extremos en los tiempos, 
       sugiriendo que una pequeña proporción de usuarios experimenta grandes demoras en ciertos pasos.
    """)

with tab6:
    st.subheader("5. Análisis de Correlaciones entre Variables")
    
    st.write("""
    En esta sección se analizan las correlaciones entre diferentes variables relacionadas con los clientes, 
    como **balance**, **edad**, **actividad** (suma de logins y llamadas a soporte), y **número de cuentas**. 
    El objetivo es descubrir relaciones significativas que puedan influir en el comportamiento de los usuarios 
    y su rendimiento en el proceso de navegación en la web.
    
    ### 5.1 Correlaciones de Pearson y Spearman
    Se utilizan dos métodos para calcular las correlaciones:
    
    - **Correlación de Pearson**: Mide la relación lineal entre dos variables continuas. 
      Un valor cercano a 1 indica una fuerte correlación positiva, mientras que un valor cercano a -1 indica 
      una fuerte correlación negativa.
    - **Correlación de Spearman**: Mide la correlación monótona entre dos variables, 
      lo que permite detectar relaciones que no son estrictamente lineales.
    """)

    # Llamar a la función para calcular y visualizar las correlaciones
    pearson_corr, spearman_corr = calculate_pearson_spearman_corr(df_filtrado)

    # Mostrar las matrices de correlación
    st.subheader("Matriz de Correlación de Pearson")
    st.write(pearson_corr)

    st.subheader("Matriz de Correlación de Spearman")
    st.write(spearman_corr)

    # Crear heatmaps para las correlaciones
    st.subheader("Visualización: Heatmaps de Correlaciones")

    fig_pearson = px.imshow(pearson_corr, text_auto=True, aspect="auto", title="Correlación de Pearson")
    st.plotly_chart(fig_pearson)

    fig_spearman = px.imshow(spearman_corr, text_auto=True, aspect="auto", title="Correlación de Spearman")
    st.plotly_chart(fig_spearman)

    # Mostrar insights sobre las correlaciones
    st.write("""
    ### Insights:
    - **Balance y Número de Cuentas**: La correlación es moderadamente positiva, sugiriendo que los clientes con más cuentas tienden a tener balances más altos.
    - **Balance y Edad**: Los clientes mayores tienden a tener balances más altos, con una correlación más fuerte en Spearman, indicando una relación no lineal.
    - **Balance y Actividad**: La correlación es baja pero positiva, sugiriendo que los clientes con balances más altos tienden a ser más activos.
    - **Actividad y Número de Cuentas**: Existe una correlación moderada, lo que significa que los clientes con más cuentas tienden a ser más activos.
    - **Edad y Actividad**: Las correlaciones son muy bajas, sugiriendo que la edad no influye significativamente en la actividad de los clientes.
    """)

with tab7:
    st.subheader("6. Pruebas Estadísticas")
    
    st.write("""
    Para validar los resultados obtenidos y confirmar si las diferencias observadas entre los grupos 
    Test (nueva versión de la web) y Control (versión antigua) son significativas, se realizan una serie de pruebas estadísticas. 
    Entre las pruebas se incluyen:
    
    - **Prueba Z**: Para comparar las tasas de finalización entre los grupos Test y Control.
    - **T-test**: Para comparar los tiempos de finalización entre los usuarios de los grupos Test y Control, así como entre usuarios lineales y no lineales.
    - **Prueba Mann-Whitney**: Para evaluar diferencias en tiempos entre los grupos cuando los datos no siguen una distribución normal.
    - **Prueba Chi-square**: Para evaluar la asociación entre las variables categóricas (Test/Control y lineal/no lineal).
    - **Cramér's V**: Para medir la fuerza de la asociación entre variables categóricas.
    """)

    st.write("### 6.1 Prueba Z para tasas de finalización")
    stat, p_value = z_test_completion_rates(df_vanguard)
    st.write(f"**Estadístico Z**: {stat}")
    st.write(f"**Valor p**: {p_value}")
    
    if p_value < 0.05:
        st.write("La diferencia en las tasas de finalización es **estadísticamente significativa**.")
    else:
        st.write("La diferencia en las tasas de finalización **NO es estadísticamente significativa**.")
    
    st.write("### 6.2 T-test para tiempos de finalización")
    step = 'confirm'
    (t_stat_lineal, p_value_lineal), (t_stat_non_lineal, p_value_non_lineal) = t_test_time_per_step(df_vanguard, step)

    st.write("**Resultados para usuarios lineales:**")
    st.write(f"**Estadístico t**: {t_stat_lineal}")
    st.write(f"**Valor p**: {p_value_lineal}")
    if p_value_lineal < 0.05:
        st.write("La diferencia en los tiempos de finalización para usuarios lineales es **estadísticamente significativa**.")
    else:
        st.write("La diferencia en los tiempos de finalización para usuarios lineales **NO es estadísticamente significativa**.")
    
    st.write("**Resultados para usuarios no lineales:**")
    st.write(f"**Estadístico t**: {t_stat_non_lineal}")
    st.write(f"**Valor p**: {p_value_non_lineal}")
    if p_value_non_lineal < 0.05:
        st.write("La diferencia en los tiempos de finalización para usuarios no lineales es **estadísticamente significativa**.")
    else:
        st.write("La diferencia en los tiempos de finalización para usuarios no lineales **NO es estadísticamente significativa**.")
    
    st.write("### 6.3 Prueba Mann-Whitney U para tiempos de finalización")
    u_statistic_variation, p_value_variation = mann_whitney_test_variation(df_vanguard)
    st.write(f"**U-statistic (Test vs Control)**: {u_statistic_variation}")
    st.write(f"**P-value (Test vs Control)**: {p_value_variation}")
    
    if p_value_variation < 0.05:
        st.write("La diferencia entre los grupos Test y Control es **estadísticamente significativa**.")
    else:
        st.write("No hay una diferencia estadísticamente significativa entre los grupos Test y Control.")
    
    u_statistic_lineal, p_value_lineal = mann_whitney_test_lineal(df_vanguard)
    st.write(f"**U-statistic (Lineal vs No Lineal)**: {u_statistic_lineal}")
    st.write(f"**P-value (Lineal vs No Lineal)**: {p_value_lineal}")
    
    if p_value_lineal < 0.05:
        st.write("La diferencia entre usuarios lineales y no lineales es **estadísticamente significativa**.")
    else:
        st.write("No hay una diferencia estadísticamente significativa entre usuarios lineales y no lineales.")
    
    st.write("### 7. Prueba Chi-square y Cramér's V")
    chi2, p_value, cramers_v = chi_square_and_cramers_v(df_vanguard)
    st.write(f"**Chi-square statistic**: {chi2}")
    st.write(f"**P-value**: {p_value}")
    st.write(f"**Cramér's V**: {cramers_v}")
    
    if p_value < 0.05:
        st.write("La asociación entre 'variation' y 'lineal' es **estadísticamente significativa**.")
    else:
        st.write("No hay una asociación estadísticamente significativa entre 'variation' y 'lineal'.")
    
    st.write(f"La fuerza de la asociación medida por **Cramér's V** es: {cramers_v}")