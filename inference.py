import torch
import spacy
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from Transformer import build_transformer

#HYPERPARAMETERS (note these exactly match the training settings)
max_src_len = 100
max_tgt_len = 100
d_model = 512
n_layers = 6
n_heads = 8
dropout = 0.1
hidden_size = 2048
alpha = 0.7
temperature = 1.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#here we build up a tokenizer and vocabularies exactly as in the training script
spacy_de = spacy.load('de_core_news_sm')  
spacy_en = spacy.load('en_core_web_sm')    

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

src_vocab = torch.load('src_vocab.pth')
tgt_vocab = torch.load('tgt_vocab.pth')

src_pad_index = src_vocab['<pad>']
tgt_pad_index = tgt_vocab['<pad>']

def create_masks(src, tgt):

    src_mask = (src != src_pad_index).unsqueeze(1).unsqueeze(2)  #[batch, 1, 1, src_seq_len]
    tgt_pad_mask = (tgt != tgt_pad_index).unsqueeze(1)             #[batch, 1, tgt_seq_len]
    tgt_seq_len = tgt.size(1)
    subseq_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).bool()  #[tgt_seq_len, tgt_seq_len]
    tgt_mask = tgt_pad_mask & subseq_mask.unsqueeze(0)             #[batch, tgt_seq_len, tgt_seq_len]
    tgt_mask = tgt_mask.unsqueeze(1)                               #[batch, 1, tgt_seq_len, tgt_seq_len]
    return src_mask, tgt_mask

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
            log_probs = torch.log_softmax(token_logits / temperature, dim=-1).squeeze(0)  # [vocab_size]
            
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
    input_sentence = "es ist uberhaupt nicht schwierig, deutsche Sätze ins Englische zu übersetzen"  #change this to any German sentence
    translation = translate_sentence(input_sentence, model, src_vocab, tgt_vocab)
    print("Input: ", input_sentence)
    print("Translation: ", translation)
