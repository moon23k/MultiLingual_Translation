from utils.data import read_data
from model.module import create_src_mask, create_trg_mask
from utils.train import Config
import sentencepiece as spm




def run(model, tokenizer, split, device, max_tokens=100):
	orig_data = read_data(f'{split}.src')
	data_seq, data_ids = [], []

	for seq in tqdm(orig_data):
	    src = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
	    src_mask = create_src_mask(src)

	    trg_indice = [tokenizer.bos_id()]

	    with torch.no_grad():
	        src = model.embedding(src.to(device))
	        enc_out = model.encoder(src.to(device), src_mask.to(device))

	        for _ in range(max_tokens):
	            trg = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0)
	            trg_mask = create_trg_mask(trg)

	            trg = model.embedding(trg.to(device))

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




if __name__ == '__main__':
	config = Config()
	model = Generator(config).to(config.device)
	model.load_state_dict(torch.load('checkpoints/gen_states.pt', map_location=config.device)['model'])
	
	tokenizer = spm.SentencePieceProcessor()
	tokenizer.load('data/vocab/spm.model')
	tokenizer.SetEncodeExtraOptions('bos:eos')

	print('Generating Train Sample datasets for Discriminator..')
	run(model, tokenizer, 'train', config.device)
	print('Train Sample Datasets are Generated!')
	print('Generating Valid Sample datasets for Discriminator..')
	run(model, tokenizer, 'valid', config.device)
	print('Valid Sample Datasets are Generated!')