import pandas as pd
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import hdbscan
import re
import nltk
import unicodedata
from nltk.stem import WordNetLemmatizer
from Levenshtein import distance as levenshtein_distance


# Télécharger les ressources nécessaires pour nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Lire le fichier parquet
input_file = '20240617_2_last_months_sirene4.parquet'
df = pd.read_parquet(input_file)
df = df[1:10000]

# Supposons que la colonne de texte s'appelle 'texte'
# Charger le modèle fastText pré-entraîné
model = fasttext.load_model('fasttext_simple_benchmark2.bin')

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

# Coalesce
print(len(df))
df['texte'] = df.activ_pr_lib_et.combine_first(df.activ_pr_lib)
# Appliquer le prétraitement à la colonne de texte
df['texte'] = df['texte'].apply(preprocess_text)

# Fonction pour obtenir l'embedding fastText moyen d'un texte
def get_text_vector(text):
    words = text.split()
    word_vectors = [model.get_word_vector(word) for word in words if word in model.words]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.get_dimension())

# Appliquer la vectorisation aux textes
df['vector'] = df['texte'].apply(get_text_vector)

# Convertir les embeddings en array pour la similarité des cosinus
vectors = np.stack(df['vector'].values)

# Calcul de la similarité des cosinus entre les textes
cosine_sim_matrix = cosine_similarity(vectors)
print(cosine_sim_matrix)

# Clustering avec HDBSCAN
clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1)
cluster_labels = clusterer.fit_predict(1 - cosine_sim_matrix)

# Définition du seuil de similarité pour DBSCAN
#epsilon = 0.95  # À ajuster en fonction de votre cas d'utilisation

# Clustering avec DBSCAN
#dbscan = DBSCAN(eps=epsilon, min_samples=2, metric='precomputed')
#cluster_labels = dbscan.fit_predict(1 - cosine_sim_matrix)

df['groupe'] = cluster_labels
print(df["groupe"].value_counts())

# Remplacer les clusters -1 par des groupes uniques pour le bruit
noise_group = df['groupe'].max() + 1
df['groupe'] = df['groupe'].apply(lambda x: x if x != -1 else noise_group)
df['groupe'] = df['groupe'].astype(str)  # Convertir en string pour l'uniformité

# Sélectionner un texte représentant par groupe
representants = df.groupby('groupe').first().reset_index()

# Afficher les résultats
print(representants)
print(len(representants))
print("Groupe 1:" + representants[representants["groupe"]=="1"]["texte"])
print(df[df["groupe"]=="1"]["texte"])
print("Groupe 5:" + representants[representants["groupe"]=="5"]["texte"])
print(df[df["groupe"]=="5"]["texte"])
print("Groupe 100:" + representants[representants["groupe"]=="100"]["texte"])
print(df[df["groupe"]=="100"]["texte"])
print("Groupe 456:" + representants[representants["groupe"]=="500"]["texte"])
print(df[df["groupe"]=="500"]["texte"])

# Sauvegarder le DataFrame résultant en parquet
#output_file = 'path/to/your/output.parquet'
#representants.to_parquet(output_file, index=False)

df.to_csv('similarity_fasttext.csv')

# Fonction pour filtrer les clusters basés sur la distance de Levenshtein
def is_text_variants(group, threshold=2):
    texts = group['texte'].tolist()
    if len(texts) < 2:
        return False
    distances = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            dist = levenshtein_distance(texts[i], texts[j])
            distances.append(dist)
    mean_distance = np.mean(distances)
    return mean_distance <= threshold  # Seuil de distance à ajuster

# Filtrer les groupes pour ne conserver que ceux avec des variantes textuelles
filtered_df = df.groupby('groupe').filter(lambda group: is_text_variants(group, threshold=4))

# Sélectionner un texte représentant par groupe
representants = filtered_df.groupby('groupe').first().reset_index()[['groupe', 'texte']]
#representants = representants.rename(columns={'texte': 'representant'})

# Joindre les représentants au DataFrame original filtré
filtered_df = filtered_df.merge(representants, on='groupe', how='left')

# Supprimer la colonne des vecteurs avant de sauvegarder
filtered_df = filtered_df.drop(columns=['vector'])

# Afficher les résultats
print(filtered_df)

# Afficher les résultats
print(representants)
print(len(representants))
print("Groupe 1:" + representants[representants["groupe"]=="1"])
print(filtered_df[filtered_df["groupe"]=="1"]["texte"])
print("Groupe 5:" + representants[representants["groupe"]=="5"]["texte"])
print(filtered_df[filtered_df["groupe"]=="5"]["texte"])
print("Groupe 100:" + representants[representants["groupe"]=="100"]["texte"])
print(filtered_df[filtered_df["groupe"]=="100"]["texte"])
print("Groupe 456:" + representants[representants["groupe"]=="456"]["texte"])
print(filtered_df[filtered_df["groupe"]=="456"]["texte"])