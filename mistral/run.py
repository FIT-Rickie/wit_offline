from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

# Load the Mistral model and tokenizer
model_name = "mistralai/mistral-7b"  # Replace this with the actual Mistral model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def sentence_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the token embeddings to get a single vector representation for the sentence
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

def sentence_similarity(sentence1, sentence2, tokenizer, model):
    embedding1 = sentence_embedding(sentence1, tokenizer, model)
    embedding2 = sentence_embedding(sentence2, tokenizer, model)
    # Compute cosine similarity
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# Example sentences
sentence1 = "Dogs are awesome."
sentence2 = "Canines are amazing."

# Compute similarity score
similarity_score = sentence_similarity(sentence1, sentence2, tokenizer, model)
print(f"Similarity score: {similarity_score}")