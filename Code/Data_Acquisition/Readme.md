# Acquisition de données depuis AtmoData API

Ce code vise à récupérer progressivement des données sur la qualité de l'air et la pollution depuis l'API AtmoData et à les stocker dans une base de données MongoDB. Les données sont acquises pour différentes régions, départements et communes en France. Elles serviront pour l'entrainement de notre modèle de prédiction.

## Prérequis

Avant d'exécuter le programme, assurez-vous d'avoir :

- [ ] Généré un token sur le site d'AtmoData et l'avoir enregistré dans un fichier nommé `token.json`.
- [ ] placé le fichier `commune_insee.xlsx`disponible depuis Sample_Data, ce fichier a été téléchargé depuis le site de l'INSEE, qui contient les codes géographiques officiels de la France depuis 2022
Nous avons pris le soin de traiter le fichier pour ne garder que les informations qui nous interesse notammment le type de commune ("TYPECOM"), le Code de la commune ("COM"), le code département ("DEP"), le code région ("REG"). Nous avons égalements supprimé certains type de communes comme les COMA et COMD qui ne nous interressé pas et appliqué des mise en forme pour faciliter son usage. Afin de pouvoir s'assurer que notre programme parcourera chaque commune une et une fois pour chaque types de données (id_data), nous avons crées une colonne supplémentaire pour chaque type de données contenant des valeurs binaires. 
- [ ] crée un fichier nommé "empty_trace_com.csv", qui, par précaution stock quelques infos sur les communes (ou régions si l'id_data =119) parcourues et qui n'ont pas donné de resultats.

## Fonctionnement

Le programme principal `general_store()` est utilisé pour stocker progressivement différents types de données (indices de qualité de l'air, pollution de l'année précédente, pollution constatée la veille, le jour même et prévue pour le lendemain, ainsi que les émissions des régions). Vous pouvez ajuster le paramètre `id_data` pour sélectionner le type de données à récupérer.

### Déclaration des fonctions

- `get_token_from_file(file_path)`: Récupère le token depuis le fichier `token.json`.
- `fetch_atmo_data(api_key, id_data, params)`: Récupère les données depuis l'API AtmoData.
- `add_data_to_database(collection, data_dict, id_data)`: Ajoute les données à une collection MongoDB.
- `write_trace_data_file(code_commune, id_data, start_date, stop_date, trace_file)`: Écrit les informations sur les zones sans données dans un fichier de trace.
- `clean_data(commune_data, id_data)`: Nettoie les données sur les communes.
- `update_status(code_insee_file, zone_code, colonne_verification)`: Met à jour le statut des communes dans le fichier Excel.
- `store_commune(api_key, id_data, start_date, commune_code, code_insee_file, trace_file)`: Stocke les données pour une commune spécifique.
- `store_departement(api_key, id_data, start_date, code_insee_file, code_dept, trace_file)`: Stocke les données pour toutes les communes d'un département.
- `store_region(api_key, id_data, start_date, region_code, code_insee_file, trace_file)`: Stocke les données pour toutes les communes d'une région.
- `fetch_and_store_region(api_key, id_data, start_date, region_code, code_insee_file, trace_file)`: Récupère et stocke les données pour une région spécifique. Utile lorsque id_data==119
-  `general_store(api_key, id_data, start_date, code_insee_file, trace_file)` : Stocke les données pour toutes les régions de la France, donc sur toutes les départements et sur toutes les communes.

### Acquisition des données

Pour récupérer les données, appelez la fonction `general_store()` avec les paramètres appropriés :

```python
api_key = "votre_clé_api"
id_data = 112  # Indice qualité de l'air
start_date = "01-01-2023"
code_insee_file = "commune_insee.xlsx"
trace_file = "empty_trace_com.csv"

general_store(api_key, id_data, start_date, code_insee_file, trace_file)
