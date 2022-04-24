import os
import time
import math
import json
import argparse
from collections import defaultdict
from model.module import Generator, Discriminator, SegGAN
from utils.train import train_epoch, eval_epoch, Config, set_seed, epoch_time
from utils.data import get_dataloader, generate_sample




def run(config):
    #set checkpoint, record path
    chk_dir = "checkpoints/"
    os.makedirs(chk_dir, exist_ok=True)
    chk_path = os.path.join(chk_dir, 'seqGAN_states.pt')
    
    record = defaultdict(list)
    record_path = os.path.join(chk_dir, 'seqGAN_record.json')
	

    #Define Tokenizer
	tokenizer = spm.SentencePieceProcessor()
	tokenizer.load('data/vocab/spm.model')


	#Load DataLoaders for Training
	train_dataloader = get_dataloader('train')
	valid_dataloader = get_dataloader('valid')


	#Load Pre-Trained Generator and Discriminator
	generator = Generator(config).to(config.device)
	generator.load_state_dict(torch.load('checkpoints/gen_states.pt', map_location=config.device)['model'])
	
	discriminator = Discriminator(config).to(config.device)
	discriminator.load_state_dict(torch.load('checkpoints/gen_states.pt', map_location=config.device)['model'])

	criterion = nn.BCELoss().to(config.device)
	optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate)


	#Adversarial Training
	print('SeqGAN Trianing')

	for epoch in range(config.n_epochs):
		start_time = time.time()
		
		train_loss = train_epoch(generator, discriminator, train_dataloader, criterion, optimizer, config.device)
		valid_loss = eval_epoch(generator, discriminator, valid_dataloader, criterion, config.device)

		end_time = time.time()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)

		print(f"Epoch: {epoch + 1}  Time: {epoch_mins}min {epoch_secs}sec")
		print(f"Train Loss: {train_loss} Valid Loss : {valid_loss}")


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



if __name__ == '__main__':
    set_seed()
    config = Config()
    run(config)