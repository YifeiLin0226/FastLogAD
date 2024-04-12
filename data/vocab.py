import pickle
import tqdm
from collections import Counter, defaultdict
import sys
import json

import numpy as np
from numpy import dot
from numpy.linalg import norm
sys.path.append("../")

class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 4
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 1
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
    
    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1, semantics = False, emb_file = None, embedding_dim = 300):
        print("Building Vocab")
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)
        self.semantics = semantics
        if self.semantics:
            with open(emb_file, "r") as f:
                self.semantic_vectors = json.load(f)
            
            self.semantic_vectors = {k: v if type(v) is list else [0] * embedding_dim for k, v in self.semantic_vectors.items()}
            self.semantic_vectors["<pad>"] = [0] * embedding_dim
            self.embedding_dim = embedding_dim
        self.mapping = {}
        

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def get_event(self, real_event, use_similar = False):
        event = self.stoi.get(real_event, self.unk_index)
        if not use_similar or event != self.unk_index:
            return event
        if self.mapping.get(real_event) is not None:
            return self.mapping.get(real_event)
        
        for train_event in self.itos[5:]:
            if train_event in self.semantic_vectors and real_event in self.semantic_vectors:
                if self.cosine_similarity(self.semantic_vectors[train_event], self.semantic_vectors[real_event]) > 0.9:
                    self.mapping[real_event] = self.stoi[train_event]
                    return self.stoi[train_event]
        
        self.mapping[real_event] = self.unk_index
        return self.unk_index

    def cosine_similarity(self, v1, v2):
        return dot(v1, v2)/(norm(v1)*norm(v2))
    

    def get_embedding(self, event):
        return self.semantic_vectors[event]
    

    def find_opposite_candidate(self, embedding):
        vocab_indices = range(5, len(self.itos))
        shuffle_indices = np.random.permutation(vocab_indices)
        for index in shuffle_indices:
            candidate_embedding = self.semantic_vectors[self.itos[index]]
            if self.cosine_similarity(candidate_embedding, embedding) < 0.2:
                return candidate_embedding
        return None
    
    def build_pseudo_anomaly(self):
        pseudo_anomaly_dict = defaultdict(list)
        for i in range(5, len(self.itos)):
            sem_i = self.semantic_vectors[self.itos[i]]
            for j in range(i, len(self.itos)):
                sem_j = self.semantic_vectors[self.itos[j]]
                if self.cosine_similarity(sem_i, sem_j) < 0.2:
                    pseudo_anomaly_dict[i].append(j)
                    pseudo_anomaly_dict[j].append(i)
        
        return pseudo_anomaly_dict

                
    

class LSTMVocab:
    def __init__(self, texts, max_size=None, min_freq=1, semantics = False, emb_file = None, embedding_dim = 300):
        print("Building Vocab")
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        

        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        self.itos = ['<unk>'] + [word for word, freq in words_and_frequencies]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_index = 0
        self.pad_index = len(self.itos)

        self.semantics = semantics
        if self.semantics:
            with open(emb_file, "r") as f:
                self.semantic_vectors = json.load(f)
            
            self.semantic_vectors = {k: v if type(v) is list else [0] * embedding_dim for k, v in self.semantic_vectors.items()}
            self.semantic_vectors["<pad>"] = [0] * embedding_dim
            self.embedding_dim = embedding_dim
        self.mapping = {}
    

    def __len__(self):
        return len(self.itos)
    

    @staticmethod
    def load_vocab(vocab_path: str) -> 'LSTMVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
        
    
    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
    

    def get_event(self, real_event, use_similar = False):
        event = self.stoi.get(real_event, self.unk_index)
        if not use_similar or event != self.unk_index:
            return event
        if self.mapping.get(real_event) is not None:
            return self.mapping.get(real_event)
        
        for train_event in self.itos[1:]:
            if train_event in self.semantic_vectors and real_event in self.semantic_vectors:
                if self.cosine_similarity(self.semantic_vectors[train_event], self.semantic_vectors[real_event]) > 0.9:
                    self.mapping[real_event] = self.stoi[train_event]
                    return self.stoi[train_event]
        
        self.mapping[real_event] = self.unk_index
        return self.unk_index

    def cosine_similarity(self, v1, v2):
        return dot(v1, v2)/(norm(v1)*norm(v2))
    

    def get_embedding(self, event):
        return self.semantic_vectors[event]
    

    def find_opposite_candidate(self, embedding):
        vocab_indices = range(1, len(self.itos))
        shuffle_indices = np.random.permutation(vocab_indices)
        for index in shuffle_indices:
            candidate_embedding = self.semantic_vectors[self.itos[index]]
            if self.cosine_similarity(candidate_embedding, embedding) < 0.2:
                return candidate_embedding
        return None
    
    def build_pseudo_anomaly(self):
        pseudo_anomaly_dict = defaultdict(list)
        for i in range(1, len(self.itos)):
            sem_i = self.semantic_vectors[self.itos[i]]
            for j in range(i, len(self.itos)):
                sem_j = self.semantic_vectors[self.itos[j]]
                if self.cosine_similarity(sem_i, sem_j) < 0.2:
                    pseudo_anomaly_dict[i].append(j)
                    pseudo_anomaly_dict[j].append(i)
        
        return pseudo_anomaly_dict
        
                