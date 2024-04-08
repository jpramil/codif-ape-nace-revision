from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import fasttext


model = SentenceTransformer("dangvantuan/sentence-camembert-large")

sentences = ["Loueur Meublé Non professionnel.",
             "Location de logements meublés non professionnel",
             "Location meublée",
             "LMNP LONGUE DUREE",
             "Location en Meublé Non Professionnel",
             "location meublée non professionnelle",
             " L'acquisition par voie d 'achat ou d'apports de tous biens mobiliers et immobiliers en pleine propriété nue propriété ou usufruit",
             "L'acquisition, l'administration et la gestion par location ou autrement de tous immeubles et biens immobiliers"
            ]

# Encode sentences
embeddings = model.encode(sentences)



# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(embeddings)

# Hierarchical clustering
linkage_matrix = linkage(similarity_matrix, method='average')

# Cluster sentences with similarity > 0.7
clusters = fcluster(linkage_matrix, 0.4, criterion='distance')

# Print clusters
for cluster_id, sentence in zip(clusters, sentences):
    print(f"Cluster {cluster_id}: {sentence}")

# Print similarity scores
for i in range(len(sentences)):
    print("Similarity scores for sentence '{}':".format(sentences[i]))
    for j in range(len(sentences)):
        if i != j:
            print("  - '{}' : {}".format(sentences[j], similarity_matrix[i][j]))
    print()

# Load pre-trained FastText model
model = fasttext.load_model('default.bin')

# Get embeddings for sentences
embeddings = [model.get_sentence_vector(sentence) for sentence in sentences]

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(embeddings)

# Print similarity scores
for i in range(len(sentences)):
    print("Fasttext: Similarity scores for sentence '{}':".format(sentences[i]))
    for j in range(len(sentences)):
        if i != j:
            print("  - '{}' : {}".format(sentences[j], similarity_matrix[i][j]))
    print()


# Hierarchical clustering
linkage_matrix = linkage(similarity_matrix, method='average')

# Cluster sentences with similarity > 0.7
clusters = fcluster(linkage_matrix, 0.4, criterion='distance')

# Print clusters
for cluster_id, sentence in zip(clusters, sentences):
    print(f"Cluster {cluster_id}: {sentence}")