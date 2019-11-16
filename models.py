import torch
import torch.nn as nn
from layers import DynamicLSTM
from torchcrf import CRF

class BiLSTMCRF(nn.Module):
    ''' The BiLSTM-CRF Model '''
    def __init__(self, embedding_matrix, opt):
        super(BiLSTMCRF, self).__init__()
        self.opt = opt
        
        WD = opt.word_emb_dim # dimension of word embeddings
        CD = opt.char_emb_dim # dimension of character embeddings
        CN = len(opt.tokenizer.vocab['char']) # number of characters in vocabulary
        C_PAD = opt.tokenizer.vocab['char'].pad_id # padding index of characters
        WHD = opt.word_hidden_dim # dimension of word-level rnn hidden state
        CHD = opt.char_hidden_dim if opt.char_emb_dim > 0 else 0 # dimension of character-level rnn hidden state
        LN = len(opt.tokenizer.vocab['label']) # number of labels in vocabulary
        
        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float)) # word embedding layer
        
        if opt.char_emb_dim > 0:
            self.char_embedding = nn.Embedding(CN, CD, padding_idx=C_PAD) # character embedding layer
            self.char_rnn = DynamicLSTM(CD, CHD, num_layers=1, batch_first=True, bidirectional=True, rnn_type='LSTM', only_use_last_hidden_state=True) # character-level RNN layer
        self.word_rnn = DynamicLSTM(WD+CHD*2, WHD, num_layers=1, batch_first=True, bidirectional=True, rnn_type='LSTM') # word-level RNN layer
        self.fc_hidden = nn.Linear(WHD*2, WHD) # fully-connected hidden layer
        self.fc_out = nn.Linear(WHD, LN) # fully-connected output layer
        self.crf = CRF(LN, batch_first=True) # CRF layer
        self.dropout = nn.Dropout(opt.dropout) # dropout
    
    def forward(self, inputs, training=True):
        text, char = inputs # words and characters in a batch, size (batch_size, word_len), (batch_size, word_len, char_len)
        word_len = torch.sum(text!=0, dim=-1) # compute the length of sentences in a batch, size (batch_size)
        word_emb = self.dropout(self.word_embedding(text)) # embed words to vectors, size (batch_size, word_len, word_emb_dim)
        if self.opt.char_emb_dim > 0:
            char_mask = torch.sum(char!=0, dim=-1, keepdims=True) # for variable length of sentences, size (batch_size, word_len)
            char_void = torch.zeros_like(char) # because the char-lstm is incompatible with padded words which length is 0, size (batch_size, word_len, char_len)
            char_void[:, :, 0] += 1 # set the length of padded words to 1
            char = torch.where(char_mask!=0, char, char_void) # update the characters, size (batch_size, word_len, char_len)
            char_len = torch.sum(char!=0, dim=-1) # compute the length of characters in a batch, size (batch_size, word_len)
            char_emb = self.dropout(self.char_embedding(char)) # embed char to vectors, size (batch_size, word_len, char_len, char_emb_dim)
            BS, WL, CL, CD = char_emb.shape # the shape of char_emb
            char_emb = char_emb.reshape(-1, CL, CD) # reshape to be compatible with char-lstm, size (batch_size*word_len, char_len, char_emb_dim)
            char_len = char_len.reshape(-1) # same as above, size (batch_size*word_len)
            char_feature = self.dropout(self.char_rnn(char_emb, char_len)) # exploit char-level RNN to extract char-level feature, outputing the last hidden state, size (num_directions, batch_size*word_len, char_hidden_dim)
            char_feature = char_feature.permute(1, 2, 0).reshape(BS, WL, -1) # reshape the char-level feature, size (batch_size, word_len, num_directions*char_hidden_dim)
            final_emb = torch.cat((word_emb, char_feature), dim=-1) # compose final word embedding, size (batch_size, word_len, word_emb_dim+num_directions*char_hidden_dim)
        else:
            final_emb = word_emb
        word_feature, _ = self.word_rnn(final_emb, word_len) # exploit word-level RNN to extract word-level feature
        output = torch.tanh(self.fc_hidden(self.dropout(word_feature))) # use a hidden layer with tanh activation
        output = self.fc_out(self.dropout(output)) # use fully-connected network to generate final representation
        if training:
            return output
        else:
            return word_len, output
    
