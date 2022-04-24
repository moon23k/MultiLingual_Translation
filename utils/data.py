import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence




def read_data(f_name, max_words=100):
    with open(f'data/daily/ids/{f_name}', 'r', encoding='utf-8') as f:
        orig_data = f.readlines()

    #cut long sentences with max_words limitation
    data = []
    for line in orig_data:
        _line = list(map(int, line.split()))
        if len(_line) > max_words:
            _line = _line[:99]
            _line.append(1) #append eos token
        data.append(_line)
    
    return data



class GenDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src = src_data
        self.trg = trg_data
    
    def __len__(self):
        return len(self.trg)
    
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]



class DisDataset(Dataset):
    def __init__(self, src_data, trg_data, label):
        self.src = src_data
        self.trg = trg_data
        self.label = label
    
    def __len__(self):
        return len(self.trg)
    
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx], self.label[idx]




def gen_collate(data_batch):    
    src_batch, trg_batch = [], []

    for batch in data_batch:
        src = torch.tensor(batch[0], dtype=torch.long)
        trg = torch.tensor(batch[1], dtype=torch.long)

        src_batch.append(src)
        trg_batch.append(trg)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=1)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=1)

    return src_batch, trg_batch




def dis_collate(data_batch):    
    src_batch, trg_batch, label_batch = [], [], []

    for batch in data_batch:
        src = torch.tensor(batch[0], dtype=torch.long)
        trg = torch.tensor(batch[1], dtype=torch.long)
        label = torch.tensor(batch[2], dtype=torch.float)

        src_batch.append(src)
        trg_batch.append(trg)
        label_batch.append(label)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=1)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=1)
    label_batch = torch.tensor(label_batch)
    
    return src_batch, trg_batch, label_batch




def get_dataloader(type, split, batch_size):
    if type == 'gen':
        src = read_data(f"{split}.src")
        trg = read_data(f"{split}.trg")

        dataset = GenDataset(src, trg)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=gen_collate, num_workers=2)
        
        return data_loader


    elif type == 'dis':
        src = read_data(f"{split}.src")
        src.extend(src)

        trg = []
        orig_trg = read_data(f'{split}.trg')
        gen_trg = read_data(f'{split}.gen')
        
        trg.extend(orig_trg)
        trg.extend(gen_trg)

        #Labeling orig_trg as 1, gen_trg as 0
        labels = [1 if i < len(orig_trg) else 0 for i in range(len(orig_trg) + len(gen_trg))]

        dataset = DisDataset(src, trg, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dis_collate, num_workers=2)

        return data_loader
