import pandas as pd
import numpy as np
import re
import nltk
import unicodedata
from nltk.stem import WordNetLemmatizer
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import pairwise_distances

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

# Fonction de clustering basé sur la distance de Levenshtein avec un seuil
def levenshtein_clustering(texts, threshold):
    n = len(texts)
    labels = -np.ones(n, dtype=int)  # Initialiser tous les labels à -1 (non assigné)
    current_label = 0

    for i in range(n):
        if labels[i] == -1:  # Si le texte n'est pas encore assigné à un cluster
            labels[i] = current_label
            for j in range(i + 1, n):
                if labels[j] == -1:  # Si le texte n'est pas encore assigné à un cluster
                    if levenshtein_distance(texts[i], texts[j]) <= threshold:
                        labels[j] = current_label
            current_label += 1

    return labels

# Appliquer le clustering avec un seuil de distance de Levenshtein
threshold = 7  # Ajuster ce seuil en fonction de votre cas d'utilisation
df['groupe'] = levenshtein_clustering(df['texte'].values, threshold)

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