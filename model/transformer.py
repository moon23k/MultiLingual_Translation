import copy, math, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        
        max_len = config.max_len
        pe = torch.zeros(max_len, config.emb_dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class Embeddings(nn.Module):
    def __init__(self, config, hidden_dim):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)

        self.pos_emb = PositionalEncoding(config)
        self.pos_dropout = nn.Dropout(config.dropout_ratio)

        self.use_fc_layer = (config.emb_dim != hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, hidden_dim)
            self.fc_dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_dropout(self.pos_emb(out))

        if not self.use_fc_layer:
            return out
        return self.fc_dropout(self.fc(out))




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.enc_hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.enc_pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )

        self.embeddings = Embeddings(config, config.enc_hidden_dim)
        self.layers = clones(layer, config.enc_n_layers)

        self.apply_link = config.model_type in ['enc_wide', 'dec_wide', 'large']
        if self.apply_link:
            self.link_layer = nn.Linear(config.enc_hidden_dim, config.dec_hidden_dim)
            self.link_dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, e_mask):

        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=e_mask)
        
        if self.apply_link:
            return self.link_dropout(self.link_layer(x))
        
        return x



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=config.dec_hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.dec_pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )

        self.embeddings = Embeddings(config, config.dec_hidden_dim)
        self.layers = clones(layer, config.dec_n_layers)

        self.apply_link = config.model_type in ['dec_wide', 'large']
        if self.apply_link:
            self.link_layer = nn.Linear(config.dec_hidden_dim, config.hidden_dim)
            self.link_dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x, memory, e_mask=None, d_mask=None):
        
        x = self.embeddings(x)        
        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask,
            )

        if self.apply_link:
            return self.link_dropout(self.link_layer(x))
        
        return x



class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')
    
    
    def pad_mask(self, x):
        return x == self.pad_id


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]    


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def forward(self, x, y):
        y, label = self.shift_y(y)

        #Masking
        e_mask = self.pad_mask(x)
        d_mask = self.dec_mask(y)
        
        #Actual Processing
        memory = self.encoder(x, e_mask)
        dec_out = self.decoder(y, memory, e_mask, d_mask)
        logit = self.generator(dec_out)
        
        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out