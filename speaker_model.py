import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import global_config
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import argparse
import wandb
import dataloader
import os
import json
import torchvision

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1)


class Speaker(pl.LightningModule):
    def __init__(self, param_args):
        super().__init__()
        self.speaker_ref = self.construct_sequential(param_args.referent)
        self.speaker_lstm = self.construct_lstm(param_args.speaker_lstm) 
        self.embedding = self.construct_embedding(param_args.embedding)
        self.embedding.requires_grad_ = False
        self.softmax = nn.Softmax(dim=2)
        self.dictionary = self.load_dictionary(param_args.dictionary)
        self.optimizer = self.construct_optimizer(param_args.optimizer, param_args.lr, param_args.weight_decay)

 
    def construct_layer(self, obj):
        if obj["model"] == "conv2d":
            in_channels = obj["in_channels"]
            out_channels = obj["out_channels"]
            kernel = obj["kernel_size"]
            stride = obj["stride"]
            return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        
        elif obj["model"] == "linear":
            in_features = obj["in_features"]
            out_features = obj["out_features"]
            return nn.Linear(in_features=in_features, out_features=out_features)

        elif obj["model"] == "flatten":
            return Flatten()
        
        else:
            raise NotImplementedError()
    
    def construct_sequential(self, lst):
        res = []
        for obj in lst:
            layer = self.construct_layer(obj)
            res += [layer]
            res += [nn.ReLU()]
        return nn.Sequential(*res)

    def construct_lstm(self, obj):
        input_size = obj["input_size"]
        hidden_size = obj["hidden_size"]
        num_layers = obj["num_layers"]
        drop_prob = obj["drop_prob"]
        if obj["batch_first"] == "True":
            batch_first = True
        else:
            batch_first = False

        return nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=drop_prob, 
            batch_first=batch_first
            )
    
    def construct_embedding(self, obj):
        token_size = obj["token_size"]
        vocab_length = obj["vocab_size"]
        self.vocab_size = vocab_length
        embedding = nn.Embedding(vocab_length, token_size)
        weights = torch.load(obj["embedding"])
        embedding.load_state_dict({"weight": weights})
        return embedding

    def load_dictionary(self, dictionary_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dictionary_path = os.path.join(script_dir, dictionary_dir)
        f_dict = open(dictionary_path).readlines()
        dictionary = json.loads(f_dict[0])
        return dictionary

    def construct_optimizer(self, optim_type, learning_rate, L2):
        if optim_type == "sgd":
            return torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=L2)
        if optim_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=L2)


    def forward(self, example):
        tkn_sentence, sentence_img = example
        ref = self.speaker_ref(sentence_img)
        length = tkn_sentence.shape[1]
        lstm_ref = ref.expand(1, length, -1)
        embed_sentence = self.embedding(tkn_sentence)
        res = torch.cat([embed_sentence, lstm_ref], dim=2)
        lstm_out, (_, _) = self.speaker_lstm(res)
        lstm_fin = torch.squeeze(lstm_out)
        return lstm_fin
    
    def inference(self, example):
        tkn_sentence, sentence_img = example
        ref = self.speaker_ref(sentence_img)
        end = False
        sentence = []
        out = torch.zeros(1, 1, self.vocab_size)
        cell = torch.zeros(1, 1, self.vocab_size)
        token = self.embedding(torch.tensor([self.dictionary["start_sentence"]]))
        hardmax = nn.hardmax(dim=0)
        while not end: 
            res = torch.cat([token, ref.expand(1, -1)])
            out, hidden, cell = self.speaker_lstm(res.expand(1, -1, -1), (hidden, cell))
            res = hardmax(torch.squeeze(out))
    

    def training_step(self, example, example_idx):
        loss = nn.CrossEntropyLoss()
        tkn_sentence, sentence_img = example
        tkn_sentence_train = tkn_sentence[:, :-1]
        tkn_sentence_test = tkn_sentence[:, 1:]
        res = tkn_sentence_train, sentence_img
        fin = self(res)
        out = torch.squeeze(fin)
        target = tkn_sentence_test
        target = torch.squeeze(tkn_sentence_test)
        batch_loss = loss(out, target)
        self.log('train_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return batch_loss
    
    def validation_step(self, example, example_idx):
        loss = nn.CrossEntropyLoss()
        tkn_sentence, sentence_img = example
        tkn_sentence_train = tkn_sentence[:, :-1]
        tkn_sentence_test = tkn_sentence[:, 1:]
        res = tkn_sentence_train, sentence_img
        fin = self(res)
        out = torch.squeeze(fin)
        target = tkn_sentence_test
        target = torch.squeeze(tkn_sentence_test)
        batch_loss = loss(out, target)
        self.log('validation_loss', batch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return batch_loss
    
    def test_step(self, example, example_idx):
        loss = nn.CrossEntropyLoss()
        out = torch.squeeze(self(example))
        target = example[0]
        batch_loss = loss(out, target)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    
    def configure_optimizers(self):
        return self.optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='config file for nlvr-pragmatics')
    parser.add_argument('--config_name', type=str, help='help="The config file name should correspond to the configs/[config_name].py', default="basic")

    args = parser.parse_args()

    param_args = global_config.get_param_args(args.config_name) 

    wandb_logger = WandbLogger(
        name=param_args.speaker_name, project="nlvr_pragmatics", entity="sidv")
  
    early_stop_callback = EarlyStopping(
        monitor='training/val_loss',
        min_delta=0.00,
        patience=40,
        verbose=False,
        mode='min'
    )

    model = Speaker(param_args)
    dictionary_dir = param_args.dictionary

    train_dir = param_args.train_dataset
    val_dir = param_args.val_dataset
    test_dir = param_args.train_dataset

    train_data = dataloader.Speaker_Dataset(dictionary_dir, train_dir, "train")
    val_data = dataloader.Speaker_Dataset(dictionary_dir, val_dir, "train")
    test_data = dataloader.Speaker_Dataset(dictionary_dir, test_dir, "test")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1,
                        shuffle=True, num_workers=6)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1,
                        shuffle=False, num_workers=6)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                        shuffle=False, num_workers=6)

    trainer = pl.Trainer(gpus=1, precision=32,
                         progress_bar_refresh_rate=5, max_epochs=60,
                         logger=wandb_logger,
                         gradient_clip_val=1)
    wandb_logger.watch(model, log='grads/all', log_freq=100)
    trainer.fit(model, train_dataloader, val_dataloader) 
    trainer.test(model, test_dataloader=test_dataloader)
    torch.save(model.Parameters, "output/" + param_args.speaker_name + ".pt")







        



    


    





        


