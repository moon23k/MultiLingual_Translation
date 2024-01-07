import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_model,
    Trainer,
    Tester,
    Generator
)



def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        self.task = args.task
        self.mode = args.mode
        self.model_type = args.model
        self.search_method = args.search        
        self.ckpt = f"ckpt/{self.task}/{self.model_type}_model.pt"
        self.tokenizer_path = f'data/{self.task}/tokenizer.json'

        self.load_config()
        self.setup_device()
        self.update_model_attrs()


    def load_config(self):
        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

    def setup_device(self):
        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda and self.mode != 'inference' else 'cpu'
        self.device = torch.device(self.device_type)


    def update_model_attrs(self):
        
        attributes = {
            'enc_n_layers': self.n_layers,
            'dec_n_layers': self.n_layers,
            'enc_n_heads': self.n_heads,
            'dec_n_heads': self.n_heads,
            'enc_hidden_dim': self.hidden_dim,
            'dec_hidden_dim': self.hidden_dim,
            'enc_pff_dim': self.pff_dim,
            'dec_pff_dim': self.pff_dim
        }

        model_type_rules = {
            'enc_wide': ['enc_hidden_dim', 'enc_pff_dim'],
            'dec_wide': ['dec_hidden_dim', 'dec_pff_dim'],
            'enc_deep': ['enc_n_layers'],
            'dec_deep': ['dec_n_layers'],
            'enc_diverse': ['enc_n_heads'],
            'dec_diverse': ['dec_n_heads'],
            'large': ['enc_hidden_dim', 'enc_pff_dim', 'dec_hidden_dim', 'dec_pff_dim',
                    'enc_n_layers', 'dec_n_layers', 'enc_n_heads', 'dec_n_heads']
        }

        model_type = self.model_type
        if model_type in model_type_rules.keys():
            rules = model_type_rules[model_type]
            for attr in rules:
                attributes[attr] *= 2
        for attr, value in attributes.items():
            setattr(self, attr, value)


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer




def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)
    tokenizer = load_tokenizer(config)


    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
    
    elif config.mode == 'inference':
        generator = Generator(config, model, tokenizer)
        generator.inference()

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-balance', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.task in ['multi-source', 'multi-targe', 'multi-lingual']
    assert args.mode in ['train', 'test', 'inference']
    assert args.balance in ['base', 'enc_deep', 'enc_wide', 'enc_diverse', 
                            'dec_deep', 'dec_wide', 'dec_diverse', 'large']
    assert args.search in ['greedy', 'beam']

    if args.mode == 'train':
        os.makedirs(f"ckpt/{args.task}", exist_ok=True)
    else:
        assert os.path.exists(f'ckpt/{args.task}/{args.balance}_model.pt')

    main(args)