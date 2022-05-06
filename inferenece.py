from utils.data import read_data
from model.module import create_src_mask, create_trg_mask
from utils.train import Config
import sentencepiece as spm
import argparse




def run(model, tokenizer, device, max_tokens=100):
	print('*** If you want to quit conversation, then type "quit". ***')
	with torch.no_grad():
		while True:
	        seq = input('\nUser >> ')
	        if seq == 'quit':
	            print('\nConversation terminated!')
	            print('------------------------------------')
	            break
	        
	        #process user input to model src tensor
	        src = tokenizer.EncodeAsIds(seq)
	        src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
	        src_mask = create_src_mask(src)

	        src = model.embedding(src)	        
	        enc_out = model.encoder(src, src_mask)
	        trg_indice = [tokenizer.bos_id()]


	        for _ in range(max_tokens):
	            trg_tensor = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0)
	            trg_mask = create_trg_mask(trg_tensor)

	            trg = model.embedding(trg_tensor)

                dec_out, _ = model.decoder(enc_out, trg, src_mask, trg_mask)
                out = model.fc_out(dec_out)

	            pred_token = out.argmax(2)[:, -1].item()
	            trg_indice.append(pred_token)

	            if pred_token == tokenizer.eos_id():
	                break
	        
	        pred_seq = trg_indice[1:]
	        pred_seq = tokenizer.Decode(pred_seq)

		    print(f"Bot >> {pred_seq}")




if __name__ == '__main__':
	config = Config()
	config.device = torch.device('cpu')
	model = Generator(config).to(config.device)
	model.load_state_dict(torch.load('checkpoints/seqGAN_states.pt', map_location=config.device)['model'])
	
	tokenizer = spm.SentencePieceProcessor()
	tokenizer.load('data/vocab/spm.model')
	tokenizer.SetEncodeExtraOptions('bos:eos')

	run(model, tokenizer, config.device)