import time
import math
import yaml
import random

import torch
import torch.nn as nn
import numpy as np



class Config(object):
    def __init__(self):
        
        files = [f"configs/{file}.yaml" for file in ['model', 'train']]

        for file in files:
            with open(file, 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            for p in params.items():
                setattr(self, p[0], p[1])


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_valid_loss = float('inf')
        self.learning_rate = 5e-4
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(attribute, ': ', value)



def init_xavier(model):
    for layer in model.named_parameters():
        if 'weight' in layer[0] and 'layer_norm' not in layer[0] and layer[1].dim() > 1:
            nn.init.xavier_uniform_(layer[1])



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




def gen_train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        src, trg = batch[0].to(device), batch[1].to(device) 

        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:].contiguous().view(-1)   
        
        pred = model(src, trg_input)
            
        pred_dim = pred.shape[-1]
        pred = pred.contiguous().view(-1, pred_dim)
        loss = criterion(pred.to(device), trg_y)
        
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item()


    return epoch_loss / len(dataloader)



def gen_eval_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            src, trg = batch[0].to(device), batch[1].to(device)   
    
            trg_input = trg[:, :-1]
            trg_y = trg[:, 1:].contiguous().view(-1)        

            pred = model(src, trg_input)

            pred_dim = pred.shape[-1]
            pred = pred.contiguous().view(-1, pred_dim)

            loss = criterion(pred, trg_y)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)





def dis_train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(dataloader):
        optimizer.zero_grad()

        src, trg, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        pred = model(src, trg)
        
        loss = criterion(pred.to(device), label)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)




def dis_eval_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for _, batch in enumerate(dataloader):    
            src, trg, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            pred = model(src, trg)

            loss = criterion(pred, label)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)




def train_epoch(generator, discriminator, dataloader, criterion, optimizer, device):
    generator.train()
    discriminator.eval()
    epoch_loss = 0

    for _, batch in enumerate(dataloader):
        optimizer.zero_grad()

        src, trg = batch[0].to(device), batch[1].to(device)
        label = torch.zeros(src.size(0), dtype=torch.float).to(device)

        sample = generator.sample(src)
        
        pred = discriminator(src, sample.to(device))
        loss = -criterion(pred.to(device), label)

        loss.backward()

        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)




def eval_epoch(generator, discriminator, dataloader, criterion, device):
    generator.eval()
    discriminator.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(dataloader):    
            src, trg = batch[0].to(device), batch[1].to(device)
            label = torch.zeros(src.size(0), dtype=torch.float).to(device)
            
            sample = generator.sample(src)
            pred = discriminator(sample.to(device))

            loss = -criterion(pred, trg.contiguous().view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)