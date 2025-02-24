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

#create a collate function that:
# 1. tokenizes the sentences
# 2. adds <bos> and <eos> tokens
# 3. converts tokens to indices
# 4. pads the sentences to the same length

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

net = build_transformer(len(src_vocab), len(tgt_vocab), max_src_len, max_tgt_len, d_model, n_layers, n_heads, dropout, hidden_size)
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr)
criterion = nn.CrossEntropyLoss()

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]  # remove <eos>
            tgt_target = tgt[:, 1:]  # remove <bos>
            src_mask, tgt_mask = create_masks(src, tgt_input)
            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, src_mask, tgt_input, tgt_mask)
            output = model.linear(dec_out)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
            total_loss += loss.item()
    model.train()  # Set back to training mode
    return total_loss / len(dataloader)

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