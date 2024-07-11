from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Chinese sentences
sentences = ["这是一个例子。", "这是另一个例子。", "这是一个不同的句子。"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform sentences into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(sentences)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(cosine_sim)