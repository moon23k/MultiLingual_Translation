import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import EncoderLayer, DecoderLayer
from .embedding import Embedding
from .block import MultiHeadAttn



def create_src_mask(src, pad_idx=1):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    src_mask.to(src.device)
    return src_mask



def create_trg_mask(trg, pad_idx=1):
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
    trg_sub_mask = torch.tril(torch.ones((trg.size(-1), trg.size(-1)))).bool()

    trg_mask = trg_pad_mask & trg_sub_mask.to(trg.device)
    trg_mask.to(trg.device)
    return trg_mask




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.n_layers = config.n_layers
        self.layer = EncoderLayer(config)


    def forward(self, src, src_mask):
        for _ in range(self.n_layers):
            src = self.layer(src, src_mask)

        return src




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.n_layers = config.n_layers
        self.layer = DecoderLayer(config)


    def forward(self, memory, trg, src_mask, trg_mask):
        for _ in range(self.n_layers):
            trg, attn = self.layer(memory, trg, src_mask, trg_mask)
        
        return trg, attn




class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.pad_idx = config.pad_idx
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx
        self.device = config.device

        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        


    def forward(self, src, trg):
        src_mask, trg_mask = create_src_mask(src), create_trg_mask(trg)
        src, trg = self.embedding(src), self.embedding(trg)

        enc_out = self.encoder(src, src_mask)
        dec_out, _ = self.decoder(enc_out, trg, src_mask, trg_mask)

        out = self.fc_out(dec_out)

        return out



    def sample(self, seq_batch, max_tokens=100):
        if seq_batch.dim() < 2:
            seq_batch = seq_batch.unsqueeze(dim=0)
        
        batch_size = seq_batch.size(0)
        samples = []
        
        with torch.no_grad():
            max_len = 0
            for src in seq_batch:
                src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
                src_mask = create_src_mask(src)

                trg_indice = [self.bos_idx]

                src_emb = self.embedding(src)
                enc_out = self.encoder(src_emb, src_mask)

                for _ in range(max_tokens):
                    trg = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0)
                    trg_mask = create_trg_mask(trg)

                    trg_emb = self.embedding(trg.to(self.device))
                    
                    dec_out, _ = self.decoder(enc_out, trg_emb, src_mask.to(self.device), trg_mask.to(self.device))
                    out = self.fc_out(dec_out)

                    pred_token = out.argmax(2)[:, -1].item()
                    trg_indice.append(pred_token)

                    if pred_token == self.eos_idx:
                        break
                
                if len(trg_indice) > max_len:
                    max_len = len(trg_indice)
                
                samples.append(trg_indice)

        samples = [x + [self.pad_idx] * (max_len - len(x)) if len(x) < max_len else x for x in samples]
        
        return torch.tensor(samples)




class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.device = config.device
        self.hidden_dim = config.hidden_dim
        self.half_dim = config.hidden_dim // 2
        self.quarter_dim = config.hidden_dim // 4
        

        self.embedding = Embedding(config)
        
        self.src_encoder = Encoder(config)
        self.trg_encoder = Encoder(config)

        self.m_attn = MultiHeadAttn(config)
        
        self.fc_1 = nn.Linear(config.hidden_dim, self.half_dim)
        self.dropout1 = nn.Dropout(config.dropout_ratio)
        
        self.fc_2 = nn.Linear(self.half_dim, self.quarter_dim)
        self.dropout2 = nn.Dropout(config.dropout_ratio)

        self.fc_out = nn.Linear(self.quarter_dim, 1)
        self.dropout3 = nn.Dropout(config.dropout_ratio)
        
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, src, trg):
        src_mask = create_src_mask(src)
        trg_mask = create_src_mask(trg)

        src, trg = self.embedding(src), self.embedding(trg) 

        src = self.src_encoder(src, src_mask)
        trg = self.trg_encoder(trg, trg_mask)

        out = self.m_attn(src, trg, trg, trg_mask)
        
        out = self.fc_1(out)
        out=  self.dropout1(F.leaky_relu(out))

        out = self.fc_2(out)
        out=  self.dropout2(F.leaky_relu(out))

        out = self.fc_out(out[:, 0, :])
        
        out = self.dropout3(F.leaky_relu(out))
        out = self.sigmoid(out)
        out = out.squeeze()
        
        return out