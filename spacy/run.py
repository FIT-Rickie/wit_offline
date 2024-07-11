import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_md')

def sentence_similarity_spacy(sentence1, sentence2):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    return doc1.similarity(doc2)

sentence1 = "Dogs are awesome."
sentence2 = "Canines are amazing."
print(f"Similarity score: {sentence_similarity_spacy(sentence1, sentence2)}")