from model import NextWordPredictor
from utils import tokeniser, index_word
import torch
import torch.nn as nn

path = 'D:/NLP/Next_word_predictor/conversion.txt'
vocab_size = 8206
embedding_dim = 100
hidden_dim = 150
model = NextWordPredictor(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load('D:/NLP/Next_word_predictor/next_word_model.pth', map_location=torch.device('cpu')))

word_to_idx, idx_to_word = index_word(path) 




''' 

   Predicting using the model

'''

input_sequence = ['good'] # Sample sequence of words
indexed_sequence = [word_to_idx[word] for word in input_sequence]
input_tensor = torch.tensor(indexed_sequence).unsqueeze(0)  # Add batch dimension
model.eval()  
with torch.no_grad():  
    output = model(input_tensor)

softmax = nn.Softmax(dim=1)
probabilities = softmax(output)
top_n = 5  # Number of top words to retrieve
top_probabilities, top_indices = torch.topk(probabilities, top_n)
top_words = [idx_to_word[idx.item()] for idx in top_indices.squeeze()]

# Display the top words and their probabilities
for word, prob in zip(top_words, top_probabilities.squeeze()):
    print(f"Word: {word}   Probabiliites:{prob}")
