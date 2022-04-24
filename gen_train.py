import os
import time
import math
import json
import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm

from utils.data import read_data, get_dataloader
from model.module import Generator, create_trg_mask, create_src_mask
from utils.train import gen_train_epoch, gen_eval_epoch, epoch_time, set_seed, init_xavier



"""
def generate_samples(model, dataloader, tokenizer, split):
    sample_ids, sample_seq = [], []
    
    for _, batch in enumerate(dataloader):
        src = batch[0]
        samples = model.sample(src).tolist()
        
        sample_seq = [tokenizer.Decode(x) for x in samples]
        sample_ids = [[str(id) for id in ids] for ids in samples]
        sample_ids = [' '.join(ids) for ids in sample_ids]


    with open(f"data/daily/ids/{split}.gen", 'w') as f:
        f.write('\n'.join(sample_ids))
    
    with open(f"data/daily/seq/{split}.gen", 'w') as f:
        f.write('\n'.join(sample_seq))
"""


def generate_samples(model, tokenizer, split, device):
    orig_data = read_data(f'{split}.src')
    data_seq, data_ids = [], []

    for seq in tqdm(orig_data):
        src = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        src_mask = create_src_mask(src)

        src = model.embedding(src.to(device))

        with torch.no_grad():
            enc_out = model.encoder(src.to(device), src_mask.to(device))
        
        trg_indice = [tokenizer.bos_id()]


        while True:
            trg = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0)
            trg_mask = create_trg_mask(trg)

            trg = model.embedding(trg.to(device))

            with torch.no_grad():
                dec_out, _ = model.decoder(enc_out, trg, src_mask.to(device), trg_mask.to(device))
                out = model.fc_out(dec_out)

            pred_token = out.argmax(2)[:, -1].item()
            trg_indice.append(pred_token)

            if pred_token == tokenizer.eos_id():
                break
        
        
        pred_seq = tokenizer.Decode(trg_indice)
        pred_ids = [str(id) for id in trg_indice]
        pred_ids = ' '.join(pred_ids)

        data_seq.append(pred_seq)
        data_ids.append(pred_ids)


    with open(f"data/daily/seq/{split}.gen", 'w') as f:
        f.write('\n'.join(data_seq))

    with open(f"data/daily/ids/{split}.gen", 'w') as f:
        f.write('\n'.join(data_ids))





def run(config):
    #set checkpoint, record path
    chk_dir = "checkpoints/"
    os.makedirs(chk_dir, exist_ok=True)
    chk_path = os.path.join(chk_dir, 'gen_states.pt')

    record = defaultdict(list)
    record_path = os.path.join(chk_dir, 'gen_record.json')


    #Define Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/vocab/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')


    #Set Generator Training Tools
    generator = Generator(config).to(config.device)
    generator.apply(init_xavier)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)
    optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate)



    #Pretrain Generator
    train_dataloader = get_dataloader('gen', 'train', config.batch_size)
    valid_dataloader = get_dataloader('gen', 'valid', config.batch_size)
    print('--- Pretraining Generator ---')
    record_time = time.time()

    for epoch in range(config.gen_epochs):
        start_time = time.time()

        train_loss = gen_train_epoch(generator, train_dataloader, criterion, optimizer, config.device)
        valid_loss = gen_eval_epoch(generator, train_dataloader, criterion, config.device)
        
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
                        'model': generator.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss}, chk_path)

        print(f"\tEpoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s")
        print(f'\t\tTrain Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}')


    train_mins, train_secs = epoch_time(record_time, time.time())
    record['train_time'].append(f"{train_mins}min {train_secs}sec")


    #save ppl score to train_record
    for (train_loss, valid_loss) in zip(record['train_loss'], record['valid_loss']):
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)

        record['train_ppl'].append(round(train_ppl, 2))
        record['valid_ppl'].append(round(valid_ppl, 2))


    #save train_record to json file
    with open(record_path, 'w') as fp:
        json.dump(record, fp)


    #Generate Data with Pre-trained Generator
    print('Generate Sample Data with Pre-trained Generator')
    genreate_samples(genreator, train_dataloader, tokenizer, 'train', config.device)
    genreate_samples(genreator, valid_dataloader, tokenizer, 'valid', config.device)



if __name__ == '__main__':    
    set_seed()
    config = Config()
    run(config)