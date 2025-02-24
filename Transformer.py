import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # use nn.Embedding to get the word embeddings from pytorch
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)



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


class FeedForwardBlock(nn.Module):
    #this is just a standard linear model like i've built plenty of times before
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))




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



#create a "connection" layer that applies the normalization step and connects the other blocks to allow for faster training
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = LayerNormalization()

    def forward(self, x, sublayer):
        #Normalize x, then pass it through a sublayer (any type), use the dropout term, and finally add x
        return x + self.dropout(sublayer(self.normalization(x)))



#in order to stack multiple encoders after embedding we create a block that does attention, normalization, feed forward layers
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



class Decoder(nn.Module):
    def __init__(self, n_layers: nn.ModuleList):
        super().__init__()
        self.n_layers = n_layers
        self.normalization = LayerNormalization()

    def forward(self, x, encoder_out, tgt_mask, src_mask):
        for layer in self.n_layers:
            x = layer(x, encoder_out, tgt_mask, src_mask)
        return self.normalization(x)



class LastLinear(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.fc(x)



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