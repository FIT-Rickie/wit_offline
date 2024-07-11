from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    # Take the average of the hidden states for the token embeddings
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def sentence_similarity_bert(sentence1, sentence2):
    embedding1 = sentence_embedding(sentence1)
    embedding2 = sentence_embedding(sentence2)
    return 1 - cosine(embedding1, embedding2)

sentence1 = "Dogs are awesome."
sentence2 = "Canines are amazing."
print(f"Similarity score: {sentence_similarity_bert(sentence1, sentence2)}")