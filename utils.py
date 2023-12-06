import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
import os

# This Function tokenises the data
def tokeniser(path):
    # Open the file in read-write mode ('+r') with UTF-8 encoding
    with open(path, '+r', encoding='utf-8') as file:
        # Read the entire content of the file
        data = file.read()

        # Convert data to lowercase and replace 'old_text' with 'new_text'
        data = data.lower()
        new_data = data.replace('old_text', 'new_text')

        # Move the file pointer to the beginning, write new_data, and truncate the rest
        file.seek(0)
        file.write(new_data)
        file.truncate()

        # Split the modified data into tokens
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", new_data)
    # Return the list of tokens
    return np.array(tokens)

def index_word(path):
    tokenized_data = tokeniser(path)
    # Create a vocabulary from the tokenized data
    vocab = set(tokenized_data)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx , idx_to_word


