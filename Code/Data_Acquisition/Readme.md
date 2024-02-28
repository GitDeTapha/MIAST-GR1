\documentclass{article}
\usepackage{listings}

\begin{document}

\title{Acquisition de données depuis AtmoData API}
\author{}
\date{}
\maketitle

Ce projet vise à récupérer des données sur la qualité de l'air et la pollution depuis l'API AtmoData et à les stocker dans une base de données MongoDB. Les données sont acquises pour différentes régions, départements et communes en France.

\section*{Prérequis}

Avant d'exécuter le programme, assurez-vous d'avoir :

\begin{itemize}
  \item Généré un token sur le site d'AtmoData et l'avoir enregistré dans un fichier nommé \texttt{token.json}.
  \item Téléchargé le fichier \texttt{commune\_insee.xlsx} depuis le site de l'INSEE, qui contient les codes géographiques officiels de la France depuis 2022.
\end{itemize}

\section*{Fonctionnement}

Le programme principal \texttt{general\_store()} est utilisé pour stocker progressivement différents types de données (indices de qualité de l'air, pollution de l'année précédente, pollution constatée la veille, le jour même et prévue pour le lendemain, ainsi que les émissions des régions). Vous pouvez ajuster le paramètre \texttt{id\_data} pour sélectionner le type de données à récupérer.

\subsection*{Déclaration des fonctions}

\begin{itemize}
  \item \texttt{get\_token\_from\_file(file\_path)}: Récupère le token depuis le fichier \texttt{token.json}.
  \item \texttt{fetch\_atmo\_data(api\_key, id\_data, params)}: Récupère les données depuis l'API AtmoData.
  \item \texttt{add\_data\_to\_database(collection, data\_dict, id\_data)}: Ajoute les données à une collection MongoDB.
  \item \texttt{write\_trace\_data\_file(code\_commune, id\_data, start\_date, stop\_date, trace\_file)}: Écrit les informations sur les zones sans données dans un fichier de trace.
  \item \texttt{clean\_data(commune\_data, id\_data)}: Nettoie les données sur les communes.
  \item \texttt{update\_status(code\_insee\_file, zone\_code, colonne\_verification)}: Met à jour le statut des communes dans le fichier Excel.
  \item \texttt{store\_commune(api\_key, id\_data, start\_date, commune\_code, code\_insee\_file, trace\_file)}: Stocke les données pour une commune spécifique.
  \item \texttt{store\_departement(api\_key, id\_data, start\_date, code\_insee\_file, code\_dept, trace\_file)}: Stocke les données pour toutes les communes d'un département.
  \item \texttt{store\_region(api\_key, id\_data, start\_date, region\_code, code\_insee\_file, trace\_file)}: Stocke les données pour toutes les communes d'une région.
  \item \texttt{fetch\_and\_store\_region(api\_key, id\_data, start\_date, region\_code, code\_insee\_file, trace\_file)}: Récupère et stocke les données pour une région spécifique.
\end{itemize}

\subsection*{Acquisition des données}

Pour récupérer les données, appelez la fonction \texttt{general\_store()} avec les paramètres appropriés :

\begin{lstlisting}[language=Python]
api_key = "votre_clé_api"
id_data = 112  # Indice qualité de l'air
start_date = "01-01-2023"
code_insee_file = "commune_insee.xlsx"
trace_file = "empty_trace_com.csv"

general_store(api_key, id_data, start_date, code_insee_file, trace_file)
\end{lstlisting}

Le programme s'arrête automatiquement lorsque le token expire. Assurez-vous de renouveler le token en cliquant sur le bouton correspondant sur le site d'AtmoData pendant l'exécution du programme.

\end{document}
