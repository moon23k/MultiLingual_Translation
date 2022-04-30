import os
import time
import math
import yaml
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from model.module import Discriminator
from utils.train import dis_train_epoch, dis_eval_epoch, epoch_time, set_seed, init_xavier, Config
from utils.data import get_dataloader, generate_sample





def run(config):
    #set checkpoint, record path
    chk_dir = "checkpoints/"
    os.makedirs(chk_dir, exist_ok=True)

    chk_path = os.path.join(chk_dir, 'dis_states.pt')
    record_path = os.path.join(chk_dir, 'dis_record.json')
    record = defaultdict(list)


    #Set Discriminator Training Tools
    discriminator = Discriminator(config).to(config.device)
    discriminator.apply(init_xavier)
    criterion = nn.BCELoss().to(config.device) #Loss의 경우 뭘로하는 게 좋을지 서칭 ㄱ
    optimizer = optim.Adam(discriminator.parameters(), lr=config.learning_rate)



    #Pretrain Discriminator
    print('Pretraining Discriminator')
    train_dataloader = get_dataloader('dis', 'train', config.batch_size)
    valid_dataloader = get_dataloader('dis', 'valid', config.batch_size)
    record_time = time.time()
    
    for epoch in range(config.dis_epochs):
        start_time = time.time()
        
        train_loss = dis_train_epoch(discriminator, train_dataloader, criterion, optimizer, config.device)
        valid_loss = dis_eval_epoch(discriminator, valid_dataloader, criterion, config.device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)


        #save training records
        record['epoch'].append(epoch+1)
        record['train_loss'].append(train_loss)
        record['valid_loss'].append(valid_loss)
        record['lr'].append(optimizer.param_groups[0]['lr'])


        #save best model
        if valid_loss < config.best_valid_loss:
            config.best_valid_loss = valid_loss
            torch.save({'epoch': epoch + 1,
                        'model': discriminator.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss}, chk_path)

        print(f"\tEpoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s")
        print(f'\t\tTrain Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}')


    train_mins, train_secs = epoch_time(record_time, time.time())
    record['train_time'].append(f"{train_mins}min {train_secs}sec")

    #save train_record to json file
    with open(record_path, 'w') as fp:
        json.dump(record, fp)



if __name__ == '__main__':    
    set_seed()
    config = Config()
    run(config)