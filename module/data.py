import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, lang_pair, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.lang_pair = lang_pair
        self.data = self.load_data(lang_pair, split)


    @staticmethod
    def load_data(lang_pair, split):
        with open(f"data/{lang_pair}/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        x = self.add_prefix(self.data[idx]['x'])
        
        x = self.tokenizer.encode(x).ids
        y = self.tokenizer.encode(self.data[idx]['y']).ids
        
        return torch.LongTensor(x), torch.LongTensor(y)


    def add_prefix(self, x):
        if self.lang_pair == 'ende':
            prefix = 'translate en to de: '
        elif self.lang_pair == 'encs':
            prefix = 'translate en to cs: '
        elif self.lang_pair == 'enru':
            prefix = 'translate en to ru: '            
        return prefix + x



class Collator(object):

    def __init__(self, pad_id):
        self.pad_id = pad_id


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)     
        
        return {'x': self.pad_batch(x_batch), 
                'y': self.pad_batch(y_batch)}


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True, 
            padding_value=self.pad_id
        )



def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(tokenizer, config.lang_pair, split), 
        batch_size=config.batch_size, 
        shuffle=split == 'train',
        collate_fn=Collator(config.pad_id),
        pin_memory=True,
        num_workers=2
    )