import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, split, lang_pair):
        super().__init__()
        self.tokenizer = tokenizer
        self.lang_pair = lang_pair
        self.data = self.load_data(split)


    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        x = self.data[idx]['en'] if self.lang_pair == 'enko' else self.data[idx]['ko']
        y = self.data[idx]['ko'] if self.lang_pair == 'enko' else self.data[idx]['en']

        x = self.add_prefix(x)
        x = self.tokenizer.encode(x).ids
        y = self.tokenizer.encode(y).ids

        return torch.LongTensor(x), torch.LongTensor(y)


    def add_prefix(self, x):
        if self.lang_pair == 'enko':
            prefix = 'translate en to ko: '
        elif self.lang_pair == 'koen':
            prefix = 'translate ko to en: '

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



def load_dataloader(config, tokenizer, split, lang_pair):
    return DataLoader(
        Dataset(tokenizer, split, lang_pair),
        batch_size=config.batch_size // 2,
        shuffle=split == 'train',
        collate_fn=Collator(config.pad_id),
        pin_memory=True,
        num_workers=2
    )

