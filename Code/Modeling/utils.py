"""
utils.py
   contient les routines communnes et définitions de variable
   Permet de ne pas surcharger le notebook Prediction.ipynb
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import requests
import json
import sys
from datetime import datetime, timezone

############################################################
#                                                          #
#        ------  Data Acquisition Area  -------            #
#                                                          #
############################################################ 
#        ------  fonctions primaires   -------           #
# Les fonctions primaires sont celles appelées dans le notebook prediction.ipynb

def get_data_by_code_zone(csv_file, code_zone):
    # Charger le fichier CSV en un DataFrame
    df = pd.read_csv(csv_file, sep=';')
    # Filtrer les lignes pour le code zone spécifié
    data = df[df['code_zone'] == int(code_zone)]
    
    # On trie 
    data_sorted = sort_by_date(data)
    return data_sorted

def get_data_by_dep(csv_file, code_dept):
    # Charger le fichier CSV en un DataFrame
    df = pd.read_csv(csv_file, sep=';')
    # Filtrer les lignes pour les codes de zone commençant par le code du département spécifié
    data = df[df['code_zone'].astype(str).str.startswith(str(code_dept))]
    # Trier
    data_sorted = sort_by_date(data)
    return data_sorted

def update_zone_info(data, id_data, code_zone):
    api_key = get_token_from_file('token.json')
    # Vérifier si des données existent pour cette zone
    if data.empty:
        print("Aucune donnée disponible pour la zone spécifiée.")
        return None
    
    # Obtenir la date la plus récente dans les données actuelles
    last_date = data['date_maj'].max()
    print("la date la plus recénte avant mise à jour :",last_date)
    # Obtenir la date actuelle
    current_date = datetime.now(timezone.utc)  # Utilisation du fuseau horaire UTC
    # Convertir last_date en format string "YYYY-MM-DD"
    formatted_last_date = last_date.strftime('%Y-%m-%d')
    # Vérifier si les données sont déjà à jour
    if last_date >= current_date:
        print("Les données sont déjà à jour.")
        return data
    # Mettre à jour les données en appelant fetch_commune
    commune_data = fetch_commune(api_key, id_data, formatted_last_date, int(code_zone))
    
    # Si des nouvelles données ont été récupérées, les ajouter à notre DataFrame
    if commune_data is not None and not commune_data.empty:
        commune_data_sorted = sort_by_date(commune_data)
        # Concaténer les nouvelles données avec les anciennes
        updated_data = pd.concat([data, commune_data_sorted], ignore_index=True)
        # Trier les données par date de mise à jour
        updated_data_sorted = sort_by_date(updated_data)
        print("Les données ont été mises à jour avec succès.")
        print(len(commune_data), "récentes observations on été ajouté")
        return updated_data_sorted
    else:
        print("Impossible de mettre à jour les données."
              +f"\n La zone ayant pour code: {code_zone} n'a pas de maj à partir de : {last_date} ")
        return data


# Cherche depuis la base mongo
def fetch_data_by_code_zone(id_data, code_zone):
    # Connexion à la base de données MongoDB
    client = MongoClient('mongodb://localhost', 27017)
    db = client['climat_france'] 
    #print("hi guys ")
    # Sélection de la collection correspondante à l'id_data
    collection_name = {
        112: 'aire_quality',
        113: 'pollution_113',
        114: 'pollution_114',
        119: 'emission_region'  
    }.get(id_data)
    
    if collection_name is None:
        print(f"Error: Unsupported id_data {id_data}.")
        return None
    
    # Récupération des données de la collection
    collection = db[collection_name]
    cursor = collection.find({'code_zone': code_zone})
    
    # Création d'un DataFrame à partir des données récupérées
    df = pd.DataFrame(list(cursor))
    
    return df
# Cherche dans un départmement depuis la base mongo 
def fetch_data_by_dep(id_data, code_dept, code_insee_file):
    # Vérification de l'id_data
    if id_data not in [112, 113, 114]:
        print("Erreur: L'id data doit-être  112, 113, ou 114.")
        return None
    
    df_combined = pd.DataFrame()
    
    # Chargement du fichier commune_insee.csv
    df_communes = pd.read_csv(code_insee_file)
    
    # Filtrage des communes pour le département spécifié
    dept_communes = df_communes[df_communes["DEP"] == code_dept]
    
    # Parcours de toutes les communes du département
    for _, commune in dept_communes.iterrows():
        commune_code = commune["COM"]
        
        # Appel à la méthode fetch_data_by_code_zone pour récupérer les informations de la commune
        commune_data = fetch_data_by_code_zone(id_data, commune_code)
        
        if commune_data is not None and not commune_data.empty:
            # Ajout des informations de la commune au DataFrame global
            df_combined = df_combined.append(commune_data, ignore_index=True)
    
    return df_combined

#        ------  fonctions secondaires   -------           #
# Les fonctions secondaires sont des routines qui servent aux primaires
#Token fourni par l'API
def get_token_from_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
        return data.get('token', None)


def fetch_data(api_key, id_data, params):

    # Construction de l'URL avec l'identifiant de la donnée
    url = f"https://admindata.atmo-france.org/api/data/{id_data}/"
    
    # Convertir les paramètres en une chaîne JSON et les ajouter à l'URL
    url += json.dumps(params)
    
    url += "?withGeom=false"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.get(url, headers=headers)
       
    if response.status_code == 200:
        data = json.loads(response.content)
        return pd.json_normalize(data['features']), response.status_code
    elif response.status_code == 401 or response.status_code == 400: 
        #sys.exit(f"Erreur: {response.status_code}") 
        return None, response.status_code
    else:
        return None, response.status_code
        #sys.exit(f"Erreur: {response.status_code}")  


def clean_data(commune_data, id_data):
    
    # Rename columns starting with "properties."
    commune_data.rename(columns=lambda x: x.replace("properties.", ""), inplace=True)
    if id_data == 112:
        commune_data.drop(columns=["geometry"], inplace=True, errors="ignore")
        commune_data.drop(columns=["type"], inplace=True, errors="ignore")
        commune_data.drop(columns=["x_wgs84"], inplace=True, errors="ignore")
        commune_data.drop(columns=["x_reg"], inplace=True, errors="ignore")
        commune_data.drop(columns=["y_reg"], inplace=True, errors="ignore")
        commune_data.drop(columns=["y_wgs84"], inplace=True, errors="ignore")
        commune_data.drop(columns=["type_zone"], inplace=True, errors="ignore")
        commune_data.drop(columns=["y_reg"], inplace=True, errors="ignore")
        commune_data.drop(columns=["epsg_reg"], inplace=True, errors="ignore")
        
    elif ((id_data == 113) or (id_data == 114)):
        # Remove the "type" column
        commune_data.drop(columns=["geometry"], inplace=True, errors="ignore")
        commune_data.drop(columns=["type"], inplace=True, errors="ignore")
    else :
        commune_data.drop(columns=["geometry"], inplace=True, errors="ignore")
        # A compléter attend de voir la structures des résultats avec une id_data == 119
        
    return commune_data


# Recupère les données d'une commune donnée ; pour les maj.
def fetch_commune(api_key, id_data, start_date, commune_code):
  # Define the parameters for the query
  params = {
      "code_zone": {"operator": "=", "value": commune_code},
      "date_ech": {"operator": ">", "value": start_date}
  }
  # Fetch data for the current commune
  commune_data, status_api = fetch_data(api_key, id_data, params)
  # Format and return data if available
  if commune_data is not None and not commune_data.empty:
    return clean_data(commune_data, id_data)
  else:
    if status_api == 400 or status_api == 401:
        sys.exit("Renouveler le token")
    return None


def sort_by_date(data):
    # Convertir les dates en type datetime sans information de décalage
    data['date_maj'] = pd.to_datetime(data['date_maj'], utc=True)
    # Trier le DataFrame en fonction de la colonne "date_maj"
    data_sorted = data.sort_values(by='date_maj')
    return data_sorted

############################################################
#                                                          #
#        ------  Data Preprossecing Area  -------          #
#                                                          #
############################################################ 

def split_train_test_set(data, test_ratio=0.3):
    # Calculer l'index de séparation entre l'ensemble d'entraînement et l'ensemble de test
    split_index = int(len(data) * (1 - test_ratio))
    
    # Diviser les données en ensembles d'entraînement et de test
    train_set = data.iloc[:split_index]
    test_set = data.iloc[split_index:]
    
    return train_set, test_set    


def remove_duplicates(df):
    """
    Remove duplicates from a DataFrame.
    
    Args:
    - df (DataFrame): Input DataFrame.
    
    Returns:
    - int: Number of duplicates removed.
    - DataFrame: DataFrame without duplicates.
    """
    # Calcul du nombre de doublons
    num_duplicates = df.duplicated().sum()
    
    # Suppression des doublons
    df_no_duplicates = df.drop_duplicates()
    
    return num_duplicates, df_no_duplicates


def display_correlations(df):
    selected_columns = df[['code_no2', 'code_o3', 'code_pm10', 'code_pm25', 'code_so2', 'code_qual']]
    
    # Calculer les corrélations entre ces colonnes
    correlations = selected_columns.corr()

    return correlations


def zscore_normalize_features(X,rtn_ms=False):
    """
    returns z-score normalized X by column
    Args:
      X : (numpy array (m,n)) 
    Returns
      X_norm: (numpy array (m,n)) input normalized by column
    """
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)

# Courbe d'apprentissage
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.show()





############################################################
#                                                          #
#        ------  Evaluation modele  Area  -------          #
#                                                          #
############################################################     
##### Fonctions des différentes métriques d'évaluation des modèles  ######  

def MAD(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def MSE(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def MAPE(y, y_pred):
    return 100 * np.mean(np.abs((y - y_pred) / y))

def RMSE(y, y_pred):
    return np.sqrt(MSE(y, y_pred))




############################################################
#                                                          #
#                 ------  Quatentaine -------              #
#                                                          #
############################################################ 
def split_train_test_features_target(df, id_data):
    
    # Tri du DataFrame par la colonne "date_ech"
    df = df.sort_values(by='date_ech')
    if (id_data == 112):
        # Sélection des variables prédictives (X) et de la variable cible (y)
        X = df[['code_no2', 'code_o3', 'code_pm10', 'code_pm25', 'code_so2']]
        y = df['code_qual']
    elif (id_data == 113 or id_data ==114):
        # Sélection des variables prédictives (X) et de la variable cible (y)
        #X = df[['code_no2', 'code_o3', 'code_pm10', 'code_pm25', 'code_so2']]
        y = df['code_pol']
    elif (id_data ==119):
        # Sélection des variables prédictives (X) et de la variable cible (y)
        #X = df[['code_no2', 'code_o3', 'code_pm10', 'code_pm25', 'code_so2']]
        y = df['code_qual']
        
    
    # Division du DataFrame en un ensemble d'entraînement (70%) et un ensemble de test (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    y_train = pd.DataFrame(y_train.values.reshape(-1, 1), columns=['code_qual'])
    y_test = y_test = pd.DataFrame(y_test.values.reshape(-1, 1), columns=['code_qual'])
    return X_train, X_test, y_train, y_test


def plot_correlation_heatmap(X, y):
    # Concaténer les features et la target
    df = pd.concat([X, y], axis=1)
    
    # Calculer les corrélations
    correlations = df.corr()
    
    # Tracer le heatmap de corrélation
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Corrélation entre les variables')
    plt.show()

def plot_r2_scores(models, X_train, y_train, X_test, y_test):
    """
    Plot R^2 scores for different models.

    Args:
    - models (list): List of tuples containing (model_name, model_instance).
    - X_train (DataFrame or array-like): Training features.
    - y_train (Series or array-like): Training target.
    - X_test (DataFrame or array-like): Test features.
    - y_test (Series or array-like): Test target.
    """
    r2_scores = []
    model_names = []

    for model_name, model in models:
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        r2_scores.append(r2)
        model_names.append(model_name)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, r2_scores, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('R^2 Score')
    plt.title('R^2 Scores for Different Models')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)  # Limit y-axis to [0, 1] for R^2 score
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

###########################"" ##########################################################""