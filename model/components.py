import copy, math, torch
import torch.nn as nn




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class SublayerConnection(nn.Module):
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



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
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)
        self.pos_emb = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.use_fc_layer = (config.emb_dim != config.hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, config.hidden_dim)


    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_emb(out)

        if self.use_fc_layer:
            return self.dropout(self.fc(out))
        return self.dropout(out)



class Decoder(nn.Module):
    def __init__(self, config):
        super(StandardDecoder, self).__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, e_mask, d_mask):

        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask,
            )

        return x



class ModelBase(nn.Module):
    def __init__(self, config):
        super(ModelBase, self).__init__()

        #Attr Setup
        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        #Output Setup
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')



    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]


    def pad_mask(self, x):
        return x == self.pad_id


    def causal_mask(self, y):
        sz = y.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)        