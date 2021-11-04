
import torch
import torch.nn as nn

class CharEmbeddings(nn.Module): 

    def __init__(self, char_size, embed_size, hidden_size):
        super(CharEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.char_embeddings = nn.Embedding(char_size, 30)
        self.projection = nn.Linear(embed_size, hidden_size, bias=False)
    
    def forward(self, X):
        X = self.char_embeddings(X)
        X = self.projection(X.view(X.size(0),X.size(1), self.embed_size))
        return X
