import time, math, json, torch
import torch.nn as nn
import torch.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau




class Trainer:
    def __init__(
        self, config, model, 
        enko_train_dataloader, 
        koen_train_dataloader, 
        enko_valid_dataloader, 
        koen_valid_dataloader
    ):

        super(Trainer, self).__init__()
        
        self.model = model
        self.enko_train_dataloader = enko_train_dataloader
        self.koen_train_dataloader = koen_train_dataloader
        self.enko_valid_dataloader = enko_valid_dataloader
        self.koen_valid_dataloader = koen_valid_dataloader
        
        self.device = config.device
        self.n_epochs = config.n_epochs
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        self.clip = config.clip
        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate        

        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=2)

        self.early_stop = config.early_stop
        self.patience = config.patience        

        self.ckpt = config.ckpt
        self.record_path = self.ckpt.replace('.pt', '.json')
        self.record_keys = ['epoch', 'train_loss', 'train_ppl', 'valid_loss', 
                            'valid_ppl', 'learning_rate', 'train_time']


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
              Train PPL: {record_dict['train_ppl']:.2f}""".replace(' ' * 14, ''))

        print(f"""  >> Valid Loss: {record_dict['valid_loss']:.3f} | \
              Valid PPL: {record_dict['valid_ppl']:.2f}\n""".replace(' ' * 14, ''))


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def train(self):
        records = []
        prev_loss, best_loss = float('inf'), float('inf')
        patience = self.patience

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
            
            #Early Stopping Process
            if self.early_stop:
                if prev_loss > val_loss:
                    patience = self.patience
            
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Ealry Stopped ---\n')
                        break

                prev_loss = val_loss

            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def pad_batch(self, batch, pad_len):
        batch_size, seq_len = batch.size()
        padded_batch = torch.full((batch_size, pad_len), self.pad_id, dtype=torch.long)
        padded_batch[:, :seq_len] = batch
        return padded_batch


    def concat_batch(self, koen_batch, enko_batch):
        x_len = max(koen_batch['x'].size(1), enko_batch['x'].size(1))
        y_len = max(koen_batch['y'].size(1), enko_batch['y'].size(1))

        koen_x_padded = self.pad_batch(koen_batch['x'], x_len)
        enko_x_padded = self.pad_batch(enko_batch['x'], x_len)

        koen_y_padded = self.pad_batch(koen_batch['y'], y_len)
        enko_y_padded = self.pad_batch(enko_batch['y'], y_len)

        x_batch = torch.cat((koen_x_padded, enko_x_padded), dim=0)
        y_batch = torch.cat((koen_y_padded, enko_y_padded), dim=0)

        concatenated_batch  = {
            'x': x_batch,
            'y': y_batch
        }

        return concatenated_batch


    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for idx, (koen_batch, enko_batch) in enumerate(zip(self.koen_train_dataloader, self.enko_train_dataloader)):
            batch = self.concat_batch(koen_batch, enko_batch)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                loss = self.model(**batch).loss
                loss = loss / self.iters_to_accumulate

            #Backward Loss
            self.scaler.scale(loss).backward()        
            
            if (idx + 1) % self.iters_to_accumulate == 0:

                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / len(self.train_dataloader), 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)  

        return epoch_loss, epoch_ppl
    

    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for koen_batch, enko_batch in zip(self.koen_valid_dataloader, self.enko_valid_dataloader):
                batch = self.concat_batch(koen_batch, enko_batch)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss = self.model(**batch).loss
                epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        

        return epoch_loss, epoch_ppl