import nltk
from nltk.corpus import wordnet as wn

# Ensure the necessary resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def sentence_similarity(sentence1, sentence2):
    sentence1_words = nltk.word_tokenize(sentence1)
    sentence2_words = nltk.word_tokenize(sentence2)
    
    # Get synsets for each word in the sentences
    synsets1 = [wn.synsets(word)[0] for word in sentence1_words if wn.synsets(word)]
    synsets2 = [wn.synsets(word)[0] for word in sentence2_words if wn.synsets(word)]
    
    # Calculate the similarity between each pair of synsets
    score, count = 0.0, 0
    for synset in synsets1:
        sim = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss)]
        if sim:
            best = max(sim)
            score += best
            count += 1
    
    # Average the score over the number of similar words
    score /= count if count else 1
    return score

sentence1 = "Dogs are awesome."
sentence2 = "puppy is amazing"
print(f"Similarity score: {sentence_similarity(sentence1, sentence2)}")