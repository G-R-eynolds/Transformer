# Building a Transformer from the ground up with PyTorch
This is a follow up to a project I've already completed building a simple feed-forward neural network using numpy. Since Transformers are all the rage right now and I have some familiarity with PyTorch as a library, this seemed like a great project to improve that knowledge and also get to grips with the fundamental theory and mathematics behind LLMs. Also, this is my first Jupyter Notebook which seemed like another tool that would be good to pick up.

The plan:
- Learn the theory and maths behind each block from articles and the original Google paper
- Note down my understanding of the methods and maths here
- Build a transformer from the ground up

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*8698hoEFnRuNtQ7vT8Lm1A.png)

### Why use a Transformer instead of an RNN?
- RNNs run very slowly for long sequences of tokens
- Using lots of hidden layers in an RNN can cause long gradient calculations to either vanish or explode due to limited precision of number representation in computation. This cause either very small or very large training updates, leading to errors
- Due to the long chain of dependencies in an RNN the "effect" of the first token in a sequence diminishes, meaning the model cannot maintain "long-range dependencies" i.e it struggles to infer context in very long sequences of text

Transformers solve all of these issues!


### The Encoder:
Transformers are split into two parts: the Encoder and Decoder which run in parallel, First up is the encoder:
The encoder produces **keys** and **values** for the encoder's multi-head attention block.
![](https://miro.medium.com/v2/resize:fit:524/format:webp/1*No33bjhlMKb0-IUqQAGH9g.png)

#### Input Embeddings:
Given a sequence of words, the sequence is first tokenized (perhaps just separating into individual words), and these tokens are then associated with a token ID (representative of their position in the vocabulary). After this the token ID is converted into a vector **emmbedding** (in this case we are using vectors of size 512). 
Note these embeddings are not fixed and in fact will be altered during the training process (this is how a transformer appears to process "meaning", or different characteristics of these words), however the token IDs are fixed.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # use nn.Embedding to get the word embeddings from pytorch
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
```

#### Positional Encoding:
The next step is positional encoding. While the embeddings aim to capture meaning, this step aims to make the model understand that words near one another are related (i.e adjective noun). This helps the model to recognise patterns in sentence structure. The formulae for positional encoding are as follows (taken directly from the original Transformer paper by Google):

$ PE(pos, 2i) = sin{\frac{pos}{1000^{\frac{2i}{d}}}} $

$ PE(pos, 2i + 1) = cos{\frac{pos}{1000^{\frac{2i}{d}}}}$

Whereby d is the dimension of the model, pos is the position of the token in the sequence, and i is the row of the embedding we are currently processing.
This means the first formula is applied to the even rows, and the second to the odd rows.



```python
class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create a zero tensor to fill in the positional embeddings
        pe = torch.zeros(seq_len, d_model)
        #create positions vector (note arange creates a vector that steps by 1 at each entry so perfect for this use case)
        position = torch.arange(start = 0, end = seq_len, dtype = torch.float).unsqueeze(1)   #this makes a vector (0.0, 1.1, 2.2 ... seq_length), then unsqueezes it to be (seq_length, 1)
        #create a "division vector" to speed up calculations of the above formulae
        div_term = torch.arange(0, d_model, 2).float() * (-math.log(1000)/d_model)     #this arange has a step of 2, baking in the 1000^2i/d here
        #apply the formulae
        pe[:,0::2] = torch.sin(position * div_term)   #note the [:,0::2 or 1] notation hits the even and odd terms of the pe vector respectively
        pe[:,1::2] = torch.cos(position * div_term)

        #add an extra dimension to make this applicable to batching
        pe = pe.unsqueeze(0)

        #want to make the module "remember" these embeddings so call the buffer method
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x is of shape [batch, pe, dim] so want to acces x(1) to get the dim of pes we require
        #from the paper the way we use positional embeddings is to add them to the original embeddings
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
```

#### Multi-Head Attention
In order to understand multi-head attention it is first helpful to understand **self-attention**, which the researchers at google adapted into multi-head attention.
Attention essentially calculates a score for each pair of words (outputted as a matrix of the same dimension as the original embedding) which can be thought of as the strength of relation of one word to another, by doing:

$ Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_k}})V $

Whereby Q,K,V are all calls to the respective input matrix Q(query), K(key), V(values)

Self-attention is permutation invariant, we expect the diagonal entries of the permutation matrix to be the largest, and certain positions can be set to $-\infty$ if we don't want them to interact

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*3d9DWq5s-36cU1kbN6VJdA.png)

Multi-head attention is different. The steps are as follows:
- Copy the input embeddings to make three copies $ Q, K, V $    (dim = (seq, d_model))
- Multiply by weight matrices $ W^Q, W^K, W^V $    (these can be tuned during training)
- Label the results $ Q', K', V' $
- Split this into h new matrices where h is the number of heads we desire (in the diagram h = 4)
- We then apply the self-attention formula to each head as above
- Label the resultant matrices $ head1, head2, head3 $; they will have dim = (seq, d_k) (d_k = d_model/h)
- This is then multiplied by a final weight matrix with dim = (h*d_k, d_model) to get the output matrix with the same size as the input

Each head can be thought of as a different aspect of a word. For example head1 could learn to relate the word as a noun, head2 as an adjective, head3 a verb and so on.


```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "Division into n_heads must be possible"
        self.d_k = d_model // n_heads   #d_k is the dim of the tensors that are run through the heads

        #initialise weight matrices
        self.w_q = nn.Linear(d_model, d_model, bias = False)  #by using a Linear layer we can speed up the calculations via PyTorch, and setting bais to False makes this just a weights matrix as we want
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)

        #initialise W_0:
        self.w_0 = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod   #this means we're defining a function that doesn't need to modify the class state
    def attention (query, key, values, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        #Transform: [Batch, n_heads, seq_len, d_k] -> [Batch, n_heads, seq_len, seq_len]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  #@ operator performs matmul so this is application of the self-attention formula
        
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9) #replaces all elements where the mask has a 0 with -1e9

        attention_scores = attention_scores.softmax(dim = -1) #apply softmax to the last dimension
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)   #just calling the PyTorch method to apply dropout here

        return (attention_scores @ values), attention_scores   #again using @ for matmul


    def forward(self, q, k, v, mask):
    #project embeddings into weight matrices:
        query = self.w_q(q)
        key = self.w_k(k)
        values = self.w_v(v)

        #need to transpose from [batch, seq_len, d_model] to [batch, seq_len, n_heads, d_k] to [batch, n_heads, seq_len, d_k]
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)  
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        #time to run attention on each
        x, self.attention_scores = MultiHeadAttention.attention(query, key, values, mask, self.dropout)
            
        #transform the output to [batch_size, seq_len, n_heads * d_k] = [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        return self.w_0(x)
```

#### ADD and Norm (Layer Normalization)
Given a batch of n items which all have features that could be embedded, follow these steps:
- Calculate the mean $\mu$ and variance $\sigma ^2$ of each item independently
- Adjust each $x_i$ in the embedding by using the following formula:
  $x_i = \frac{x_i - \mu _i}{\sqrt{\sigma_i^2 + \epsilon}}$
- This is then multiplied by paramaters $\alpha$, $\lambda$(multiplicative) or $\beta$(additive)
- $\epsilon$ is added in the denominator so it doesn't approach zero


```python
class LayerNormalization(nn.Module):
    def __init__(self, eps = 10e-6):
        super().__init__()
        self.eps = eps
        #want alpha and beta to be trainable so we call the Paramater method which tells PyTorch to train these
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))  #both of these create a 1d tensor with 1 element i.e [0]

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)   #-1 is the last dimension, makes sure we are taking the mean of the values of x
        std = x.std(dim = -1, keepdim = True)
        norm = self.alpha * (x -mean)/torch.sqrt(std ** 2 + self.eps) + self.beta
        return norm
```


```python
#create a "connection" layer that applies the normalization step and connects the other blocks to allow for faster training
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = LayerNormalization()

    def forward(self, x, sublayer):
        #Normalize x, then pass it through a sublayer (any type), use the dropout term, and finally add x
        return x + self.dropout(sublayer(self.normalization(x)))
```


```python
#in order to be able to stack multiple blocks, need to build a block
class EncoderBlock(nn.Module):
    def __init__ (self, attention_block :MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])  #creates a list with 2 residual connection modules

    def forward(self, x ,mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, mask))  #first connection block takes output of multi-head attention so we feed x into attention here
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))  #second one feeds into the feed forward layer
        return x
```


```python
#now time for the main encoder class:
class Encoder(nn.Module):
    def __init__(self, n_layers: nn.ModuleList):
        super().__init__()
        self.n_layers = n_layers
        self.normalization = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.n_layers:
            x = layer(x, mask)
        return self.normalization(x)
```

### The Decoder:
That completes the encoder, now it's time to build the decoder:
![](https://miro.medium.com/v2/resize:fit:544/format:webp/1*Hjin7_ljwxRcmvojICizig.png)

#### Masked Multi-Head Attention
The idea here is to make the model **casual**, meaning the output at a given position only depends on the preceding words. This means we have to stop the transformer seeing future words, which is achieved my **masking** during the multi-head attention process.
Specifically this involves setting all entries above the diagonal to $-\infty$.
This block then produces the **querys** for the decoder's main multi-head attention block.

#### Feed Forward
The output of the multi-head attention block followed by ADD-and-norm is a tensor which can be fed into a standard feed-forward neural network. This usually consists of two fully connected layers with ReLU activation functions to allow the model to learn non-linear behaviour. The output is always a tensor of the same shape as the original input.


```python
class FeedForwardBlock(nn.Module):
    #this is just a standard linear model like i've built plenty of times before
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```


```python
#same idea with Decoder; want to be able to stack multiple after embedding so we define an attention-norm-feedforward block first
class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])  #creates a list with 3 residual connection modules

    def forward(self, x, encoder_out, mask1, mask2):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask1))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_out, encoder_out, mask2))
        x = self.residual_connections[2](x, lambda x: self.feed_forward(x))
        return x
```


```python
class Decoder(nn.Module):
    def __init__(self, n_layers: nn.ModuleList):
        super().__init__()
        self.n_layers = n_layers
        self.normalization = LayerNormalization()

    def forward(self, x, encoder_out, tgt_mask, src_mask):
        for layer in self.n_layers:
            x = layer(x, encoder_out, tgt_mask, src_mask)
        return self.normalization(x)
```

#### Output
This is just a standard linear layer, allowing any loss to be used in training.


```python
class LastLinear(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.fc(x)
```


```python
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, src_pos: PositionalEmbeddings, tgt_pos: PositionalEmbeddings, 
                 last_linear: LastLinear):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.last_linear = last_linear   #self-explanatory

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, enc_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_out, tgt_mask, src_mask)

    def linear(self, x):
        return self.last_linear(x)
    #this whole thing is made very nice by the classes defined above
```


```python
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len:int, d_model: int = 512, n_layers: int = 6,
                      n_heads:int = 8, dropout: float = 0.1, hidden_size: int = 2048):
    #first step is to make embedding layers:
    src_embeddings = InputEmbeddings(d_model, src_vocab_size)
    tgt_embeddings = InputEmbeddings(d_model, tgt_vocab_size)

    #now we make pos embed layers:
    src_pos = PositionalEmbeddings(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbeddings(d_model, tgt_seq_len, dropout)

    #create the encoder blocks:
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        feed_forward = FeedForwardBlock(d_model, hidden_size, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    #create the decoder blocks:
    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        feed_forward = FeedForwardBlock(d_model, hidden_size, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    #create encoder:
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    #create decoder:
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #create last linear layers
    last_layer = LastLinear(d_model, tgt_vocab_size)

    #finally time to define the Transformer in full:
    transformer = Transformer(encoder, decoder, src_embeddings, tgt_embeddings, src_pos, tgt_pos, last_layer)

    #initialise paramaters with Xavier intialisation (helps to avoid vanishing gradients)    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)    

    return transformer
```

### Training
Before implementing a Music Transformer, I though it prudent to train this one to do machine translation as it was originally developed for by google. To do this I will use the multi30k dataset from torchtext, and pyTorch to implement tokenisation, batching etc.


```python
import torch
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from Transformer import build_transformer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import spacy

#HYPERPARAMETERS
max_src_len = 100
max_tgt_len = 100
d_model = 512
n_layers = 6
n_heads = 8
dropout = 0.1
hidden_size = 2048
batch_size = 32
lr = 0.0001
n_epochs = 10

device = torch.device('cuda')
```

First step is to load a dataset and a tokenizer. Here I'm using spaCy and the Multi30k dataset from torchtext:


```python
spacy_de = spacy.load('de_core_news_sm')  #german tokenizer
spacy_en = spacy.load('en_core_web_sm')     #english tokenizer

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]  #creates list of german tokens

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]  #creates list of english tokens

special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]  #symbols for unknows, padding, beginning of sequence, end of sequence

#note we are translating from german to english so src is german and tgt is english
def yield_tokens(data_iter, language):   #function to generate tokenized sentences in either language
    for src_sample, tgt_sample in data_iter:
        if language == 'de':
            yield tokenize_de(src_sample)
        else:
            yield tokenize_en(tgt_sample)

train_iter = Multi30k(split='train', language_pair=('de', 'en'))  #training data, note Multi30k returns tuples of (german, english) sentences
src_vocab = build_vocab_from_iterator(yield_tokens(train_iter, 'de'), specials=special_symbols)  #builds german vocab
src_vocab.set_default_index(src_vocab["<unk>"])  #set default index for unknown words

train_iter = Multi30k(split='train', language_pair=('de', 'en'))  #reset training data as it is an iterator
tgt_vocab = build_vocab_from_iterator(yield_tokens(train_iter, 'en'), specials=special_symbols)  #builds english vocab
tgt_vocab.set_default_index(tgt_vocab["<unk>"])  #set default index for unknown words
```

Now I create a custom collate function for the dataloader to use which will:
- tokenize the sentences from the dataset (note these are already cleaned)
- add <bos> and <eos> tokens
- converts tokens to indices
- pads the sentences


```python
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_tokens =["<bos>"] + tokenize_de(src_sample) + ["<eos>"]  #add <bos> and <eos> tokens
        tgt_tokens =["<bos>"] + tokenize_en(tgt_sample) + ["<eos>"]

        src_tokens = src_tokens[:max_src_len]
        tgt_tokens = tgt_tokens[:max_tgt_len]
        
        src_indices = src_vocab(src_tokens)  #convert tokens to indices
        tgt_indices = tgt_vocab(tgt_tokens)

        src_batch.append(torch.tensor(src_indices, dtype=torch.long))  #add to batch
        tgt_batch.append(torch.tensor(tgt_indices, dtype=torch.long))
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab["<pad>"])  #pad the sentences
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab["<pad>"])

    return src_batch, tgt_batch
```

Now we create masks for the training data to be fed into the encoder and decoder blocks. In the encoder these will simply mask out any <pad> tokens to avoid error, and in the decoder we ensure casuality by masking out the top half of the matrix as explained earlier.


```python
train_dataset = list(Multi30k(split='train', language_pair=('de', 'en')))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#need to know when it starts padding in order to create masks for training
src_pad_index = src_vocab['<pad>']
tgt_pad_index = tgt_vocab['<pad>']

def create_masks(src, tgt):
    src_mask = (src != src_pad_index).unsqueeze(1).unsqueeze(2) #makes a matrix of 0s where there is padding, 1s elsewhere
    tgt_pad_mask = (tgt != tgt_pad_index).unsqueeze(1) 
    tgt_seq_len = tgt.size(1)
    subseq_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device = device)).bool() #"tril" makes a lower triangular matrix, so this has 1s below the diagonal and 0s above to ensure casuality in MMA
    tgt_mask = tgt_pad_mask & subseq_mask.unsqueeze(0) #takes into account padding also
    tgt_mask = tgt_mask.unsqueeze(1)
    return src_mask, tgt_mask
```

I also implemented an evaluation function that will be run after each epoch to get another input as to how well the model is performing after each epoch.


```python
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]  #remove <eos>
            tgt_target = tgt[:, 1:]  #remove <bos>
            src_mask, tgt_mask = create_masks(src, tgt_input)
            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, src_mask, tgt_input, tgt_mask)
            output = model.linear(dec_out)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
            total_loss += loss.item()
    model.train()  #set back to training mode
    return total_loss / len(dataloader)
```

All that's left is to define a standard training loop using Adam and CrossEntropyLoss in pyTorch:


```python
net = build_transformer(len(src_vocab), len(tgt_vocab), max_src_len, max_tgt_len, d_model, n_layers, n_heads, dropout, hidden_size)
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr)
criterion = nn.CrossEntropyLoss()

for epoch in range (n_epochs):
    epoch_loss = 0.0
    for batch in train_dataloader:
        src, tgt = batch
        src = src.to(device)  #transformer expects shape (seq_len, batch_size)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]  #remove <eos> token
        tgt_target = tgt[:, 1:]  #remove <bos> token
        src_mask, tgt_mask = create_masks(src, tgt_input)
        optimizer.zero_grad()
        enc_out = net.encode(src, src_mask)
        dec_out = net.decode(enc_out, src_mask,tgt_input, tgt_mask)
        output = net.linear(dec_out)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))  #reshape to 2D tensor
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    valid_loss = evaluate(net, train_dataloader)
    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_dataloader)} Validation Loss: {valid_loss}")

torch.save(net.state_dict(), 'transformer_weights.pth') #save the weights for inference
torch.save(src_vocab, 'src_vocab.pth')
torch.save(tgt_vocab, 'tgt_vocab.pth')
```


```python
Epoch 1 Loss: 1.7731107343255226
Epoch 2 Loss: 0.4258847597251259
Epoch 3 Loss: 0.21841126423670018
Epoch 4 Loss: 0.13542658144807343
Epoch 5 Loss: 0.09036654730090232
Epoch 6 Loss: 0.06228175041057342
Epoch 7 Loss: 0.043085280171253584
Epoch 8 Loss: 0.028818972463706197
Epoch 9 Loss: 0.017904086781564542
Epoch 10 Loss: 0.008983496685775229
```

The rapid convergence is proof of the power of the transformer architecture, however, it turned out this was due to some flaws in the training methodology and some overfitting, which resulted in an essentially useless model desipte the very low loss.


```python
Epoch 25 Loss: 4.1449397439095505e-05
```

#### Inference
I wanted to run some tests on the trained model to get a feel for how well it ran, so I created a simple inference script to run inputs through the decoder of the trained model. Note the code below uses the same vocabularies as the training loop so I omitted that section here.


```python
model = build_transformer(len(src_vocab), len(tgt_vocab),
                          max_src_len, max_tgt_len,
                          d_model, n_layers, n_heads,
                          dropout, hidden_size)
model = model.to(device)

#load saved weights after training
model.load_state_dict(torch.load('transformer_weights.pth', map_location=device))
model.eval()

#translation function using beam search
def translate_sentence(sentence, model, src_vocab, tgt_vocab, max_len=100, beam_size=5):
    #preprocess input
    tokens = ["<bos>"] + tokenize_de(sentence) + ["<eos>"]
    src_indices = torch.tensor(src_vocab(tokens), dtype=torch.long).unsqueeze(0).to(device)  # [1, src_seq_len]
    
    src_mask = (src_indices != src_pad_index).unsqueeze(1).unsqueeze(2)
    enc_out = model.encode(src_indices, src_mask)
    
    #define special tokens
    bos_token = tgt_vocab["<bos>"]
    eos_token = tgt_vocab["<eos>"]
    
    #initialize beam with a tuple (sequence, cumulative_log_prob)
    beams = [([bos_token], 0.0)]
    
    for _ in range(max_len):
        new_beams = []
        #if all beams already end with <eos>, stop expanding
        if all(seq[-1] == eos_token for seq, score in beams):
            break
        
        #expand each beam candidate
        for seq, score in beams:
            if seq[-1] == eos_token:
                #do not expand if already ended
                new_beams.append((seq, score))
                continue

            #prepare target sequence tensor for the current beam
            tgt_seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)  # [1, current_seq_len]
            _, tgt_mask = create_masks(src_indices, tgt_seq)
            dec_out = model.decode(enc_out, src_mask, tgt_seq, tgt_mask)
            output = model.linear(dec_out)  # [1, current_seq_len, vocab_size]
            
            #consider only the last time step
            token_logits = output[:, -1, :]  # [1, vocab_size]
            log_probs = torch.log_softmax(token_logits, dim=-1).squeeze(0)  # [vocab_size]
            
            #get top beam_size token probabilities for current beam
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
            
            #create new beam candidates by appending each top token
            for k in range(beam_size):
                new_seq = seq + [topk_indices[k].item()]
                # Use a length penalty (alpha is a hyperparameter, e.g., 0.7):
                length = len(seq) + 1  # +1 for the new token being added
                penalty = (5 + length) ** alpha / (5 + 1) ** 0.7  # example from GNMT
                new_score = (score + topk_log_probs[k].item()) / penalty
                new_beams.append((new_seq, new_score))
        
        #keep only the top beam_size candidates across all expansions
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
    
    #select the best candidate from the beams (highest score)
    best_seq, best_score = beams[0]
    #remove <bos> and trailing <eos> if present
    if best_seq[0] == bos_token:
        best_seq = best_seq[1:]
    if best_seq and best_seq[-1] == eos_token:
        best_seq = best_seq[:-1]
    
    #convert token indices to words using tgt_vocab.get_itos()
    translated_tokens = [tgt_vocab.get_itos()[i] for i in best_seq]
    return " ".join(translated_tokens)


if __name__ == "__main__":
    input_sentence = "Ein junges Mädchen sitzt auf einer Bank und hält ein rotes Eis am Stiel."  #change this to any German sentence
    translation = translate_sentence(input_sentence, model, src_vocab, tgt_vocab)
    print("Input: ", input_sentence)
    print("Translation: ", translation)
```


```python
Input:  Ein junges Mädchen sitzt auf einer Bank und hält ein rotes Eis am Stiel.
Translation:  pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool 
pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool pool
```

At this stage the model initially just output <eos> regardless of the input prompt. I tried implementing a length penalty to encourage it to output longer strings, which led to outputs such as the one above. After some considered research I decided this might be tue to exposure bias occuring because of the use of teacher forcing during training. This is essentially a phenomenon whereby because the model is always exposed to the ground truth at every step during training, it then collapses when those truths are absent during inference. To combat this is I implemented the following changes:


```python
#added functionality to the training loop to feed the model increasing amounts of it's own predictions as training went on:
for epoch in range(n_epochs):
    epoch_loss = 0.0
    #gradually decrease teacher forcing probability
    teacher_forcing_ratio = max(0.3, 0.95 ** epoch)

    #as before

    use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio)

    if use_teacher_forcing:
            dec_out = net.decode(enc_out, src_mask, tgt_input, tgt_mask)
            output = net.linear(dec_out)
        else:
            #mix teacher forcing and model predictions
            seq_len = tgt_input.size(1)
            batch_size = tgt_input.size(0)
            outputs = torch.zeros(batch_size, seq_len, len(tgt_vocab)).to(device)
            
            #always start with <bos> token
            decoder_input = tgt_input[:, 0].unsqueeze(1)
            
            for t in range(seq_len):
                #create appropriate mask for current sequence length
                _, step_mask = create_masks(src, decoder_input)
                
                #get decoder output for current step
                step_output = net.decode(enc_out, src_mask, decoder_input, step_mask)
                pred = net.linear(step_output)
                
                #store prediction for loss calculation
                if t < seq_len:
                    outputs[:, t:t+1] = pred[:, -1:, :]
                
                #get next token (either from ground truth or prediction)
                if t < seq_len - 1:
                    # mix ground truth and predicted tokens for next timestep
                    use_ground_truth = (torch.rand(batch_size, 1).to(device) < teacher_forcing_ratio)
                    
                    #get predicted token
                    next_token = pred[:, -1].argmax(dim=-1).unsqueeze(1)
                    true_token = tgt_input[:, t+1].unsqueeze(1)
                    mixed_token = torch.where(use_ground_truth, true_token, next_token)  #where use_ground_truth is true, use true_token, else use next_token
                    
                    decoder_input = torch.cat([decoder_input, mixed_token], dim=1)
            
            output = outputs
#last bit as above
```


```python
#added label smoothing in the training loop to try and prevent overfitting
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```


```python
#during inference, added a temperature variable to try and increase variability of the beams produced
temperature = 1.2

 log_probs = torch.log_softmax(token_logits / temperature, dim=-1).squeeze(0)
```

The result of this tweaking was increased training times (due to the fact that when not using teacher forcing, the model processes token-by-token sequentially rather than in parallel), and much higher loss: 


```python
Epoch 1 Loss: 2.829222243702267 Teacher forcing: 1.00
Epoch 2 Loss: 1.77390589708793 Teacher forcing: 0.95
Epoch 3 Loss: 1.6888313067393108 Teacher forcing: 0.90
Epoch 4 Loss: 1.6705951677530577 Teacher forcing: 0.86
Epoch 5 Loss: 1.7488125560039196 Teacher forcing: 0.81
Epoch 6 Loss: 1.6804233560751396 Teacher forcing: 0.77
Epoch 7 Loss: 1.7289672420989586 Teacher forcing: 0.74
Epoch 8 Loss: 1.7149874652759505 Teacher forcing: 0.70
Epoch 9 Loss: 1.7352698361860484 Teacher forcing: 0.66
Epoch 10 Loss: 1.7673742050898509 Teacher forcing: 0.63
```

However, when running inference I was able to get outputs (although not perfect):


```python
Input:  Ein junges Mädchen sitzt auf einer Bank und hält ein rotes Eis am Stiel.
Translation:  A young girl is sitting on a bench and holding a red ice cream .  

Input:  Die Katze schläft auf dem Sofa
Translation:  The cat cat is on the couch .

Input:  Ich habe gestern einen interessanten Film gesehen
Translation:  I I I to be with some sort . . .

Input:  Der Klimawandel ist eine der größten Herausforderungen unserer Zeit
Translation:  The pitcher is a a the the the . . .

Input:  Ich freue mich darauf, dich bald wiederzusehen
Translation:  I on the track , , and the . . .
```

Compared to the google translate outputs of the same sentences:
- A young girl sits on a bench and holds a red popsicle.
- The cat sleeps on the sofa
- I saw an interesting film yesterday
- Climate change is one of the greatest challenges of our time
- I look forward to seeing you again soon

We see that when given sentences similar to those contained in the Multi30k dataset (descriptions of images, scenarios) it provides somewhat meaningful, or even very accurate, translations. However, when giving more complex or idiomatic speech, it struggles greatly.
This is likely due simply to the limited scope of the dataset (30,000 sentences is really very small), and the model size (I had to keep it limited so I could train it locally at a reasonable speed). Both of these could be upscaled, however as this project is more of a proof of concept to check that my architecture works correctly, I'm happy to leave it with these results.