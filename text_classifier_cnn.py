# CNN for text classification


from collections import defaultdict
import time
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np



# The CNN Model
class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags, BATCH_SIZE):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        # torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        self.embedding.weight = torch.nn.Parameter(weights, requires_grad=True)
        # Conv 1d
        self.window_size = [3,4,5]
        self.conv_1d = torch.nn.ModuleList([torch.nn.Conv1d(in_channels = emb_size, out_channels = num_filters , kernel_size=win, stride=1, padding=1, dilation=1, groups=1, bias=True) for win in self.window_size])
        self.relu = torch.nn.ReLU()
        self.projection_layer = torch.nn.Linear(in_features=num_filters*len(self.window_size), out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        self.drop = torch.nn.Dropout(0.8)

    # forward pass
    def forward(self, words):
        emb = self.embedding(words)                 # create an embedding matrix
        emb = emb.permute(1, 2, 0)                  # dont do this in minibatch

        relul = []
        for i, l in enumerate(self.conv_1d):     
            h = self.conv_1d[i](emb)                # Convolution
            h = h.max(dim=2)[0]                     # max pooling
            h = self.relu(h)                        # ReLU
            relul.append(h)
        h = torch.cat(relul, dim = 1)               # Concatenate
        out = self.drop(h)                          # Dropout
        out = self.projection_layer(out)            # Fully connected layer     

        return out






# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

# List of stop words
cached_stopwords = ['all', 'just', "don't", 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'o', 'don', 'hadn', 'herself', 'll', 'had', 'should', 'to', 'only', 'won', 'under', 'ours', 'has', "should've", "haven't", 'do', 'them', 'his', 'very', "you've", 'they', 'not', 'during', 'now', 'him', 'nor', "wasn't", 'd', 'did', 'didn', 'this', 'she', 'each', 'further', "won't", 'where', "mustn't", "isn't", 'few', 'because', "you'd", 'doing', 'some', 'hasn', "hasn't", 'are', 'our', 'ourselves', 'out', 'what', 'for', "needn't", 'below', 're', 'does', "shouldn't", 'above', 'between', 'mustn', 't', 'be', 'we', 'who', "mightn't", "doesn't", 'were', 'here', 'shouldn', 'hers', "aren't", 'by', 'on', 'about', 'couldn', 'of', "wouldn't", 'against', 's', 'isn', 'or', 'own', 'into', 'yourself', 'down', "hadn't", 'mightn', "couldn't", 'wasn', 'your', "you're", 'from', 'her', 'their', 'aren', "it's", 'there', 'been', 'whom', 'too', 'wouldn', 'themselves', 'weren', 'was', 'until', 'more', 'himself', 'that', "didn't", 'but', "that'll", 'with', 'than', 'those', 'he', 'me', 'myself', 'ma', "weren't", 'these', 'up', 'will', 'while', 'ain', 'can', 'theirs', 'my', 'and', 've', 'then', 'is', 'am', 'it', 'doesn', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', "shan't", 'shan', 'needn', 'haven', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'm', 'yours', "you'll", 'so', 'y', "she's", 'the', 'having', 'once']


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])





train = list(read_dataset("topicclass/topicclass_train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass/topicclass_valid.txt"))
test = list(read_dataset("topicclass/topicclass_test.txt"))
i2t = dict((v,k) for k, v in t2i.iteritems())
print(i2t)

nwords = len(w2i)
ntags = len(t2i)
print('loaded data')

# Define the model hyperparameters
EMB_SIZE = 300
WIN_SIZE = 10
FILTER_SIZE = 100
BATCH_SIZE = int(128)

# Read the word embeddings
fname = 'GoogleNews-vectors-negative300.bin'
vectors = {}
with open(fname, "rb") as f:
    a = f.readline()
    n_words, embed_size = a.split()
    n_words = int(n_words)
    embed_size = int(embed_size)            
    for i in range(0,n_words):
        word = []
        ch = f.read(1)
        while(ch != ""):
            if ch != '\n':
                if ch == ' ':
                    word = ''.join(word)
                    vectors[word] = np.fromstring(f.read(4 * embed_size), dtype='float32') # Read the next 4 bytes i.e. the embedding vector
                    break
                else:
                    word.append(ch) 
            ch = f.read(1)                  # Reads 1 byte i.e. 1 character

# Randomly initialize if word in vocab does not have a pre trained embedding
embed = np.random.uniform(-0.25, 0.25, (nwords, EMB_SIZE))
for word, vec in vectors.items():
    if word in w2i:
        embed[w2i[word]] = vec
weights = torch.FloatTensor(embed)
print('loaded embeddings')

# initialize the model
model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags, BATCH_SIZE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00005)

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()






# Training and Validation
print('start training')
for ITER in range(22):
    # Perform training
    model.train()
    random.shuffle(train)                       # shuffle the data
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()

    no_of_batches = int(len(train)/BATCH_SIZE) # calculate number of mini batches
    for i in range(0, no_of_batches):
        start = i*BATCH_SIZE
        end = start+BATCH_SIZE
        sent = train[start:end]                 # batchsize x sentence length

        words = [torch.tensor(x[0]).type(type) for x in sent]
        tag = [x[1] for x in sent]
        tag_tensor = torch.tensor(tag).type(type)
        # Padding 
        words_tensor = pad_sequence(words)      # batchsize x nmax

        scores = model(words_tensor)            # Calculate scores
        predict = torch.argmax(scores,dim = 1)  # Predict by finding the incdex of the maximum score     
        correct = torch.nonzero(torch.eq(predict, tag_tensor)).size(0)  # finds the number of non-zero elements on equating predicted tags with actual tags
                                                                        # i.e. essentially counts the number of correct predictions
        train_correct += correct
        my_loss = criterion(scores, tag_tensor) # Calculate cross entropy loss
        train_loss += my_loss.item()  

        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
        ITER, train_loss / no_of_batches, train_correct / len(train), (time.time() - start)))
    

    # Perform validation
    val_correct = 0.0
    val_loss = 0.0
    val_pred = []
    model.eval()
    no_of_batches = int(len(dev)/BATCH_SIZE)

    for i in range(0, no_of_batches):
        start = i*BATCH_SIZE
        end = start+BATCH_SIZE
        sent = dev[start:end]                   #batchsize x sentence length

        words = [torch.tensor(x[0]).type(type) for x in sent]
        tag = [x[1] for x in sent]
        tag_tensor = torch.tensor(tag).type(type)
        # Padding 
        words_tensor = pad_sequence(words)

        scores = model(words_tensor)
        predict = torch.argmax(scores,dim = 1)
        # val_pred.append(predict.tolist())
        correct = torch.nonzero(torch.eq(predict, tag_tensor)).size(0)
        val_correct += correct
        my_loss = criterion(scores, tag_tensor)
        val_loss += my_loss.item()
    print("iter %r: val loss=%.4f, val acc=%.4f" % (ITER, val_loss/no_of_batches, val_correct / len(dev)))





######### final predictions ###############
######### VAL ##############
val_correct = 0.0
val_loss = 0.0
val_pred = []
model.eval()
no_of_batches = int(len(dev)/BATCH_SIZE)
for i in range(0, no_of_batches):
    start = i*BATCH_SIZE
    end = start+BATCH_SIZE
    sent = dev[start:end]                   #batchsize x sentence length

    words = [torch.tensor(x[0]).type(type) for x in sent]
    tag = [x[1] for x in sent]
    tag_tensor = torch.tensor(tag).type(type)
    # Padding 
    words_tensor = pad_sequence(words)

    scores = model(words_tensor)
    predict = torch.argmax(scores,dim = 1)
    val_pred.append(predict.tolist())
    correct = torch.nonzero(torch.eq(predict, tag_tensor)).size(0)
    val_correct += correct
    my_loss = criterion(scores, tag_tensor)
    val_loss += my_loss.item()
print("iter %r: val loss=%.4f, val acc=%.4f" % (ITER, val_loss/no_of_batches, val_correct / len(dev)))
val_pred = np.array(val_pred).flatten()

######## TEST ##########
model.eval()
no_of_batches = int(len(test)/BATCH_SIZE)
test_pred = []
for i in range(0, no_of_batches):
    start = i*BATCH_SIZE
    end = start+BATCH_SIZE
    sent = test[start:end] #batchsize x sentence length

    words = [torch.tensor(x[0]).type(type) for x in sent]
    tag = [x[1] for x in sent]
    tag_tensor = torch.tensor(tag).type(type)
    # Padding 
    words_tensor = pad_sequence(words)
    scores = model(words_tensor)
    predict = torch.argmax(scores,dim = 1)
    test_pred.append(predict.tolist())
test_pred = np.array(test_pred).flatten()

print('\n\n')
print('val predictions')
for i in range(0,len(val_pred)):
    print(i2t[val_pred[i]])

print('\n\n')
print('test predictions')
for j in range(0,len(test_pred)):
    print(i2t[test_pred[j]])







# In[ ]:




