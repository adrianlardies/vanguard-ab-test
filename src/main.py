import pandas as pd
import numpy as np

# Función para cargar los datos desde archivos CSV
def cargar_datos():
    df_final_demo = pd.read_csv('../data/raw/df_final_demo.txt')
    df_final_experiment_clients = pd.read_csv('../data/raw/df_final_experiment_clients.txt')
    df_pt_1 = pd.read_csv('../data/raw/df_pt_1.txt')
    df_pt_2 = pd.read_csv('../data/raw/df_pt_2.txt')
    
    # Concatenar df_pt_1 y df_pt_2
    df_pt = pd.concat([df_pt_1, df_pt_2], ignore_index=True)
    
    return df_final_demo, df_final_experiment_clients, df_pt

# Función para limpiar los datos eliminando duplicados y valores nulos
def limpiar_datos(df_final_demo, df_final_experiment_clients, df_pt):
    df_final_demo = df_final_demo.dropna().drop_duplicates()
    df_final_experiment_clients = df_final_experiment_clients.dropna()
    df_pt = df_pt.drop_duplicates()
    
    return df_final_demo, df_final_experiment_clients, df_pt

# Función para filtrar por client_id comunes en las tablas
def filtrar_por_clientes_comunes(df_final_demo, df_final_experiment_clients, df_pt):
    client_ids_demo = set(df_final_demo['client_id'])
    client_ids_experiment = set(df_final_experiment_clients['client_id'])
    client_ids_pt = set(df_pt['client_id'])
    
    # Intersección de client_ids en común
    client_ids_comunes = client_ids_demo.intersection(client_ids_experiment).intersection(client_ids_pt)

    # Filtrar las tablas por client_ids en común
    df_final_demo = df_final_demo[df_final_demo['client_id'].isin(client_ids_comunes)]
    df_final_experiment_clients = df_final_experiment_clients[df_final_experiment_clients['client_id'].isin(client_ids_comunes)]
    df_pt = df_pt[df_pt['client_id'].isin(client_ids_comunes)]

    return df_final_demo, df_final_experiment_clients, df_pt

# Convertir tipos de datos en df_final_demo
def convertir_tipos_datos(df_final_demo, df_pt):
    # Convertir columnas en df_final_demo a enteros
    df_final_demo['clnt_tenure_yr'] = df_final_demo['clnt_tenure_yr'].astype(int)
    df_final_demo['clnt_tenure_mnth'] = df_final_demo['clnt_tenure_mnth'].astype(int)
    df_final_demo['num_accts'] = df_final_demo['num_accts'].astype(int)
    df_final_demo['calls_6_mnth'] = df_final_demo['calls_6_mnth'].astype(int)
    df_final_demo['logons_6_mnth'] = df_final_demo['logons_6_mnth'].astype(int)
    
    # Convertir date_time a formato datetime en df_pt
    df_pt['date_time'] = pd.to_datetime(df_pt['date_time'])
    
    return df_final_demo, df_pt

# Renombrar columnas y fusionar datasets
def renombrar_columnas_y_fusionar(df_final_experiment_clients, df_final_demo, df_pt):
    # Renombrar las columnas de df_pt
    df_pt = df_pt.rename(columns={'process_step': 'step'})
    
    # Renombrar columnas en df_final_demo
    df_final_demo = df_final_demo.rename(columns={
        'clnt_tenure_yr': 'tenure_years',
        'clnt_tenure_mnth': 'tenure_months',
        'clnt_age': 'age',
        'gendr': 'gender',
        'num_accts': 'num_accounts',
        'bal': 'balance',
        'calls_6_mnth': 'calls_6_months',
        'logons_6_mnth': 'logons_6_months'
    })
    
    # Renombrar df_final_experiment_clients a df_final y renombrar su columna 'Variation' a 'variation'
    df_final_experiment_clients = df_final_experiment_clients.rename(columns={'Variation': 'variation'})
    
    # Fusionar df_final_demo y df_final_experiment_clients en un único DataFrame (df_final)
    df_final = pd.merge(df_final_experiment_clients, df_final_demo, on='client_id', how='inner')
    
    return df_final, df_pt

# Función para calcular medias y percentiles de las métricas clave
def calcular_medias_percentiles(df_final):
    mean_tenure = df_final['tenure_years'].mean()
    mean_balance = df_final['balance'].mean()
    mean_num_accounts = df_final['num_accounts'].mean()
    mean_activity = (df_final['calls_6_months'] + df_final['logons_6_months']).mean()

    percentile_90_tenure = df_final['tenure_years'].quantile(0.90)
    percentile_90_balance = df_final['balance'].quantile(0.90)
    percentile_90_num_accounts = df_final['num_accounts'].quantile(0.90)
    percentile_90_activity = (df_final['calls_6_months'] + df_final['logons_6_months']).quantile(0.90)
    
    return mean_tenure, mean_balance, mean_num_accounts, mean_activity, percentile_90_tenure, percentile_90_balance, percentile_90_num_accounts, percentile_90_activity

# Función para clasificar los clientes en diferentes grupos
def clasificar_clientes(df_final, medias, percentiles):
    mean_tenure, mean_balance, mean_num_accounts, mean_activity = medias
    percentile_90_tenure, percentile_90_balance, percentile_90_num_accounts, percentile_90_activity = percentiles
    
    df_above_mean = df_final[
        (df_final['tenure_years'] > mean_tenure) &
        (df_final['balance'] > mean_balance) &
        (df_final['num_accounts'] > mean_num_accounts) &
        ((df_final['calls_6_months'] + df_final['logons_6_months']) > mean_activity)
    ]

    df_top_10 = df_final[
        (df_final['tenure_years'] >= percentile_90_tenure) &
        (df_final['balance'] >= percentile_90_balance) &
        (df_final['num_accounts'] >= percentile_90_num_accounts) &
        ((df_final['calls_6_months'] + df_final['logons_6_months']) >= percentile_90_activity)
    ]

    df_final['group'] = 'General'
    df_final.loc[df_final['client_id'].isin(df_top_10['client_id']), 'group'] = 'Top 10%'
    df_final.loc[
        (df_final['tenure_years'] > mean_tenure) &
        (df_final['balance'] > mean_balance) &
        (df_final['num_accounts'] > mean_num_accounts) &
        ((df_final['calls_6_months'] + df_final['logons_6_months']) > mean_activity) &
        (df_final['group'] == 'General'),
        'group'
    ] = 'Above Mean'

    df_final['activity'] = df_final['calls_6_months'] + df_final['logons_6_months']
    df_final['age_group'] = pd.cut(df_final['age'], bins=[0, 35, 55, 100], labels=['Young', 'Middle-aged', 'Senior'])
    df_final['balance_category'] = pd.qcut(df_final['balance'], q=3, labels=['Low', 'Medium', 'High'])
    df_final['activity_category'] = pd.qcut(df_final['activity'], q=3, labels=['Low', 'Medium', 'High'])
    df_final['tenure_category'] = pd.cut(df_final['tenure_years'], bins=[0, 5, 10, 20, 50], labels=['0-5', '6-10', '11-20', '21+'])

    return df_final

# Función para analizar pasos en df_pt
def analizar_pasos(df_pt):
    df_pt = df_pt.sort_values(by=['client_id', 'visit_id', 'date_time'])
    
    step_order = {'start': 1, 'step_1': 2, 'step_2': 3, 'step_3': 4, 'confirm': 5}
    df_pt['step_order'] = df_pt['step'].map(step_order)
    
    df_pt['step_diff'] = df_pt.groupby(['client_id', 'visit_id'])['step_order'].diff().fillna(0)
    df_pt['non_linear'] = df_pt['step_diff'].apply(lambda x: x < 0)
    df_pt['step_repeat_count'] = df_pt.groupby(['client_id', 'visit_id', 'step']).cumcount() + 1
    df_pt['time_diff'] = df_pt.groupby(['client_id', 'visit_id'])['date_time'].diff().fillna(pd.Timedelta(seconds=0))
    df_pt['time_diff'] = df_pt['time_diff'].dt.total_seconds()
    df_pt['total_time_in_step'] = df_pt.groupby(['client_id', 'visit_id', 'step_repeat_count'])['time_diff'].cumsum()

    max_repeats = df_pt.groupby(['client_id', 'visit_id'])['step_repeat_count'].max() <= 2
    no_retrocessions = ~(df_pt.groupby(['client_id', 'visit_id'])['step_diff'].apply(lambda x: (x == -1).any()))
    
    def follows_correct_sequence_with_repeats(group):
        steps = group['step_order'].tolist()
        correct_sequence = [1, 2, 3, 4, 5]
        valid_steps = []
        for step in correct_sequence:
            count = steps.count(step)
            if count == 0:
                return False
            valid_steps.extend([step] * min(count, 2))
        return steps == valid_steps

    correct_order = df_pt.groupby(['client_id', 'visit_id']).apply(follows_correct_sequence_with_repeats)
    is_lineal = max_repeats & no_retrocessions & correct_order
    df_pt = df_pt.merge(is_lineal.rename('lineal'), on=['client_id', 'visit_id'])

    return df_pt

# Función para construir el DataFrame final
def construir_df_vanguard(df_final, df_pt):
    # Fusionar df_pt con df_final basado en 'client_id'
    df_vanguard = pd.merge(df_pt, df_final.drop_duplicates(subset='client_id'), on='client_id', how='left')

    # Eliminar columnas duplicadas si existen
    df_vanguard = df_vanguard.drop(columns=['variation_y', 'lineal_y'], errors='ignore')
    df_vanguard = df_vanguard.rename(columns={'variation_x': 'variation', 'lineal_x': 'lineal'})

    return df_vanguard
