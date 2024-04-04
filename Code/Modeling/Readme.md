# Modélisation

Dans cette section du projet, nous explorons et modélisons les données relatives à la qualité et à la pollution de l'air en France. Voici un aperçu des fichiers et des configurations clés utilisés dans ce processus.

## Structure des fichiers

- `utils.py`: Ce fichier contient des fonctions utilitaires qui servent à alléger le notebook `Prediction.ipynb`. Il est important que `utils.py` soit situé dans le même répertoire que `Prediction.ipynb`. Si ce n'est pas le cas, veuillez ajuster le chemin d'importation des fonctions dans `Prediction.ipynb` dans l'instruction `from utils import *`.

- `Prediction.ipynb`: Le notebook principal utilisé pour la prédiction et l'analyse des données de qualité et de pollution de l'air.

## Données

Les données utilisées dans ce projet sont structurées comme suit :

- `aire_quality = "./data/climat_france.aire_quality.csv"`: Chemin vers le fichier contenant les données sur l'indice de qualité de l'air.
- `polution_113 = "./data/climat_france.pollution_113.csv"`: Chemin vers le premier fichier de données sur la pollution.
- `polution_114 = "./data/climat_france.pollution_114.csv"`: Chemin vers le second fichier de données sur la pollution.

Assurez-vous que ces fichiers sont disponibles dans un répertoire `/data/` situé dans le même répertoire que les codes. Ces fichiers, ou une partie de leurs données, sont disponibles dans la section "Sample_Data".

Pour ceux qui ont accès localement à notre base de données MongoDB, une fonction est prévue pour récupérer les données directement.

## Mise à jour des données

De plus, une fonction de mise à jour permet d'obtenir les nouvelles informations pour une zone donnée, en faisant une requête d'accès via API à ATMO. Assurez-vous que le fichier de token nécessaire pour l'accès API est bien généré (voir la section acquisition pour plus de détails).

La mise à jour a pour but de requérir de nouvelles informations qui sont postérieures à la date la plus récente disponible dans les fichiers CSV ou dans la base de données.

---

Cette section vise à fournir une vue d'ensemble claire et concise des composantes de la modélisation dans notre projet d'exploration de la qualité et de la pollution de l'air en France. Pour toute question ou clarification, n'hésitez pas à ouvrir un issue sur GitHub.
