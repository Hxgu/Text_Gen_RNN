import codecs
import os
import numpy as np
from collections import Counter
from six.moves import cPickle
import jieba

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading Input file..."
class TextLoader():
    def __init__(self, data_dir, data_file, batch_size, seq_length, encoding='utf-8'):
    #Initialize private variables
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

    #Create input_file, vocab_file and tensor file
        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("Found no pre-existing vocab and tensor file, reading from text")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("Previously processed vocab and tensor filed found, reading from them")
            self.load_preprocessed(vocab_file, tensor_file)

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        seg_list = list(jieba.cut(data))
        vocab_counter = Counter(seg_list)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.words, _ = zip(*count_pairs)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocess(self, input_file, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))