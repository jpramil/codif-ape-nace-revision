import pandas as pd
import numpy as np
import re
import nltk
import unicodedata
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering

# Télécharger les ressources nécessaires pour nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Lire le fichier parquet
input_file = '20240617_2_last_months_sirene4.parquet'
df = pd.read_parquet(input_file)
df = df[1:10000]

# Coalesce des colonnes de texte
df['texte'] = df.activ_pr_lib_et.combine_first(df.activ_pr_lib)

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Convertir en minuscules
    text = text.lower()

    # Supprimer les accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Supprimer la ponctuation
    text = re.sub(r'\W', ' ', text)
    
    # Tokenisation
    tokens = nltk.word_tokenize(text)
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Reconstituer le texte
    text = ' '.join(tokens)
    
    return text

# Appliquer le prétraitement à la colonne de texte
df['texte'] = df['texte'].apply(preprocess_text)

# Utilisation de RapidFuzz pour calculer les distances de Levenshtein
def rapidfuzz_distance(text1, text2):
    return 100 - fuzz.ratio(text1, text2)

# Créer une matrice de distances basée sur RapidFuzz
texts = df['texte'].values
n = len(texts)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        dist = rapidfuzz_distance(texts[i], texts[j])
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

# Remplacer les valeurs infinies par une grande valeur (par exemple, 100)
dist_matrix[np.isinf(dist_matrix)] = 100

# Clustering avec AgglomerativeClustering en utilisant la matrice de distances pré-computée
clustering_model = AgglomerativeClustering(
    n_clusters=None,  # Ajuster selon votre besoin
    linkage='ward',  # Méthode d'agrégation
    distance_threshold=2,
    #affinity='precomputed'  # Utilisation de la matrice de distances pré-calculée
)
cluster_labels = clustering_model.fit_predict(dist_matrix)

df['groupe'] = cluster_labels

# Sélectionner un texte représentant par groupe
representants = df.groupby('groupe').first().reset_index()

# Afficher les résultats
print(representants)
print(len(representants))
print("Groupe 1:", representants[representants["groupe"]==1]["texte"].values)
print(df[df["groupe"]==1]["texte"].values)
print("Groupe 5:", representants[representants["groupe"]==5]["texte"].values)
print(df[df["groupe"]==5]["texte"].values)
print("Groupe 100:", representants[representants["groupe"]==100]["texte"].values)
print(df[df["groupe"]==100]["texte"].values)
print("Groupe 500:", representants[representants["groupe"]==500]["texte"].values)
print(df[df["groupe"]==500]["texte"].values)


df.to_csv('similarity.csv')