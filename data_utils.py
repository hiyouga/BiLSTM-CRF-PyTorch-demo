import os
import pickle
import numpy as np
from torch.utils.data import Dataset

def parse_conll2003(fname, lower=True): # parse conll2003 NER dataset
    fdata = open(fname, 'r', encoding='utf-8').read()
    samples = list()
    for sent in fdata.strip().split('\n\n'):
        try:
            chars, tokens, labels = list(), list(), list()
            for line in sent.strip().split('\n'):
                token, _, _, label = line.strip().split()
                char = list(token) # characters should not be lowered
                token = token.lower() if lower else token
                chars.append(char)
                tokens.append(token)
                labels.append(label)
            samples.append((chars, tokens, labels))
        except Exception:
            print('Failed in ', sent)
    return samples

class Vocab(object):
    ''' vocabulary for the datasets '''
    def __init__(self, vocab_list, add_pad=True, add_unk=True):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad: # pad_id should always be zero (for mask)
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._vocab_dict[self.pad_word] = self.pad_id
            self._length += 1
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._vocab_dict[self.unk_word] = self.unk_id
            self._length += 1
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w
    
    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, idx):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(idx, self.unk_word)
        return self._reverse_vocab_dict[idx]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    

class Tokenizer(object):
    ''' transform text to indices '''
    def __init__(self, word_vocab, char_vocab, label_vocab, lower):
        self.vocab = {
            'word': word_vocab,
            'char': char_vocab,
            'label': label_vocab
        }
        self.maxlen = {
            'word': 50,
            'char': 10,
            'label': 50
        }
        self.lower = lower
    
    @classmethod
    def from_files(cls, fnames, lower=True):
        all_chars, all_tokens, all_labels = set(), set(), set() # preparing for vocabulary
        for fname in fnames:
            samples = parse_conll2003(fname, lower=lower)
            for chars, tokens, labels in samples:
                for char in chars:
                    all_chars.update(char)
                all_tokens.update(tokens)
                all_labels.update(labels)
        word_vocab = Vocab(all_tokens)
        char_vocab = Vocab(all_chars)
        label_vocab = Vocab(all_labels, add_pad=False, add_unk=False)
        return cls(word_vocab, char_vocab, label_vocab, lower)
    
    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x
    
    def to_sequence(self, tokens, vocab_name, reverse=False, padding='post', truncating='post'):
        sequence = [self.vocab[vocab_name].word_to_id(t) for t in tokens]
        pad_id = self.vocab[vocab_name].pad_id if hasattr(self.vocab[vocab_name], 'pad_id') else 0
        maxlen = self.maxlen[vocab_name]
        if reverse:
            sequence.reverse()
        return self.pad_sequence(sequence, pad_id, maxlen, padding=padding, truncating=truncating)
    

class MyDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer):
        data_file = os.path.join('dats', os.path.split(fname)[-1].replace('.txt', '.dat')) # cache for dataset
        if os.path.exists(data_file):
            print(f"loading dataset: {data_file}")
            dataset = pickle.load(open(data_file, 'rb'))
        else:
            print('building dataset...')
            dataset = self.read_data(fname, tokenizer)
            pickle.dump(dataset, open(data_file, 'wb'))
        self._dataset = dataset
    
    @staticmethod
    def read_data(fname, tokenizer):
        dataset = list()
        samples = parse_conll2003(fname, tokenizer.lower)
        for chars, tokens, labels in samples:
            char_list = list()
            word_len = min(len(chars), tokenizer.maxlen['word'])
            for i in range(word_len):
                char_list.append(tokenizer.to_sequence(chars[i], 'char'))
            for i in range(tokenizer.maxlen['word']-word_len):
                char_list.append(tokenizer.to_sequence([], 'char'))
            chars = np.stack(char_list)
            tokens = tokenizer.to_sequence(tokens, 'word')
            labels = tokenizer.to_sequence(labels, 'label')
            dataset.append({'text': tokens, 'char': chars, 'label': labels})
        return dataset
    
    def __getitem__(self, index):
        return self._dataset[index]
    
    def __len__(self):
        return len(self._dataset)
    

def _get_chunk(tags, length): # get chucks of the labels
    result = dict()
    index = 0
    split_tag = lambda tag: ('O', None) if tag == 'O' else tag.split('-', maxsplit=1)
    while index < length:
        prefix_p, suffix_p = split_tag(tags[index])
        if prefix_p == 'B':
            k = index + 1
            while k < length:
                prefix_k, suffix_k = split_tag(tags[k])
                if prefix_k == 'I' and suffix_k == suffix_p:
                    k += 1
                else:
                    break
            result[str(index)] = (index, k, suffix_p) # (start index, end index, entity type)
            index = k
        else:
            index += 1
    return result

def compute_score(lengths, predicts, labels, tokenizer):
    n_pred, n_true, n_correct = 0, 0, 0
    for k in range(lengths.shape[0]):
        length = lengths[k]
        pred_k = _get_chunk([tokenizer.vocab['label'].id_to_word(t) for t in predicts[k][:length]], length)
        true_k = _get_chunk([tokenizer.vocab['label'].id_to_word(t) for t in labels[k][:length]], length)
        n_pred += len(pred_k)
        n_true += len(true_k)
        n_correct += sum([1 if i in pred_k and pred_k[i]==v else 0 for i, v in true_k.items()]) # match chucks in predicted labels and output labels
    P = n_correct / n_pred if n_pred else 0
    R = n_correct / n_true if n_true else 0
    F1 = 2 * P * R / (P + R) if P + R else 0
    return F1

def build_tokenizer(fnames):
    data_file = os.path.join('dats', 'tokenizer.dat') # cache for tokenizer
    if os.path.exists(data_file):
        print(f"loading tokenizer: {data_file}")
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        print('building tokenizer...')
        tokenizer = Tokenizer.from_files(fnames)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer

def _load_wordvec(embed_file, word_dim, vocab=None):
    with open(embed_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        word_vec['<pad>'] = np.zeros(word_dim).astype('float32') # embedding vector for <pad> is always zero
        for line in f:
            tokens = line.rstrip().split()
            if (len(tokens)-1) != word_dim:
                continue
            if tokens[0] == '<pad>' or tokens[0] == '<unk>':
                continue
            if vocab is None or vocab.has_word(tokens[0]):
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        return word_vec

def build_embedding_matrix(vocab, word_dim=300):
    data_file = os.path.join('dats', 'embedding_matrix.dat') # cache for embedding matrix
    embed_file = os.path.join('..', 'glove', 'glove.840B.300d.txt') # glove pre-trained word embeddings
    if os.path.exists(data_file):
        print(f"loading embedding matrix: {data_file}")
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), word_dim)).astype('float32') # sample from U(-0.25,0.25)
        word_vec = _load_wordvec(embed_file, word_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix
