import os
import time
# import yaml
import json
import random
import torch
import torch.optim as optim
from collections import namedtuple
import math
import sys

from data_loader import get_loader, get_gcn_loader, get_gcn_lstm_loader, \
    get_gtr_loader, get_gcn_gtr_loader
from torch_geometric.data import DataLoader


from LSTMModel import LSTMModel
from GCNModel import GCNModel
from GTRModel import GTRModel2
from utils import config
from GCNGTR2Model import GCNGTR2Model
import numpy as np


device_id=0
SEED = 43
# SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# torch.cuda.set_device(device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

model_type = config.model_type


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - elapsed_time * 60)
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, epoch, id2word):
    if model_type == 'gcn_lstm':
        model.encoder_gcn.train()
        model.encoder_lstm.train()
    elif model_type == 'gcn_gtr2':
        model.encoder_gcn.train()
        model.encoder_gtr.train()
    elif model_type == 'gcn_gtr2_lstm':
        model.encoder_gcn.train()
        model.encoder_gtr.train()
        model.encoder_lstm.train()
    elif model_type == 'gcn_gtr2_ae':
        model.encoder_gcn.train()
        model.encoder_gtr.train()
        model.encoder_ae.train()
    else:
        model.encoder.train()
    model.decoder.train()
    model.reduce_state.train()

    return model.forward(iterator, device, optimizer, epoch)


def evaluate(model, iterator):
    if model_type == 'gcn_gtr2':
        model.encoder_gcn.eval()
        model.encoder_gtr.eval()
    else:
        model.encoder.eval()
    model.decoder.eval()
    model.reduce_state.eval()

    with torch.no_grad():
        epoch_loss = model.eval(iterator, device)

    return epoch_loss/len(iterator)


def sort_beams( beams):
    return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


def beam_search(model, iterator, word2id, id2word, rplc_dict=None):
    if model_type=='gcn_gtr2':
        model.encoder_gcn.eval()
        model.encoder_gtr.eval()
    else:
        model.encoder.eval()

    model.decoder.eval()
    model.reduce_state.eval()

    translate_results = []
    f = open(config.test_write_to, 'w+')

    if rplc_dict==None:
        with torch.no_grad():
            model.beam_search(f, iterator,device,word2id,id2word)
    else:
        with torch.no_grad():
            model.beam_search(f, iterator,device,word2id,id2word,rplc_dict)
    f.close()


def main():
    with open(config.word2id_vocab_file, 'r', encoding = 'utf-8') as fd:
        word2id = json.load(fd)
    with open(config.id2word_vocab_file, 'r', encoding = 'utf-8') as fd:
        id2word = json.load(fd)


    if model_type == 'bilstm':
        model = LSTMModel()
        params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
                 list(model.reduce_state.parameters())

        data_loader_train = get_loader(config.train_data_file, word2id, max_enc_steps=config.max_enc_steps, max_dec_steps=config.max_dec_steps, batch_size=config.batch_size)
        data_loader_val = get_loader(config.val_data_file, word2id, max_enc_steps=config.max_enc_steps, max_dec_steps=config.max_dec_steps,batch_size=config.batch_size)
        data_loader_test = get_loader(config.test_data_file, word2id, max_enc_steps=config.max_enc_steps, max_dec_steps=config.max_dec_steps,batch_size=1, shuffle=False)
    elif model_type == 'gcn':
        model = GCNModel()
        params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.reduce_state.parameters())

        data_loader_train = get_gcn_loader(config.train_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                       max_dec_steps=config.max_dec_steps, batch_size=config.batch_size)
        data_loader_val = get_gcn_loader(config.val_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                     max_dec_steps=config.max_dec_steps, batch_size=config.batch_size)
        data_loader_test = get_gcn_loader(config.test_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                      max_dec_steps=config.max_dec_steps, batch_size=1, shuffle=False)
    elif model_type == 'gtr' or model_type == 'gtr2':
        model = GTRModel2()
        # model = GTRModel()
        params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.reduce_state.parameters())

        data_loader_train = get_gtr_loader(config.train_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                       max_dec_steps=config.max_dec_steps, batch_size=config.batch_size)
        data_loader_val = get_gtr_loader(config.val_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                     max_dec_steps=config.max_dec_steps, batch_size=config.batch_size)
        data_loader_test = get_gtr_loader(config.test_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                      max_dec_steps=config.max_dec_steps, batch_size=1, shuffle=False)

    elif model_type == 'gcn_gtr2':
        model = GCNGTR2Model()
        params = list(model.encoder_gcn.parameters()) + list(model.encoder_gtr.parameters()) + \
                 list(model.decoder.parameters()) + list(model.reduce_state.parameters())

        data_loader_train = get_gcn_gtr_loader(config.train_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                       max_dec_steps=config.max_dec_steps, batch_size=config.batch_size)
        data_loader_val = get_gcn_gtr_loader(config.val_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                     max_dec_steps=config.max_dec_steps, batch_size=config.batch_size)
        data_loader_test = get_gcn_gtr_loader(config.test_data_file, word2id, max_enc_steps=config.max_enc_steps,
                                      max_dec_steps=config.max_dec_steps, batch_size=1, shuffle=False)

        num_nodes_batch = []
        for i, (src_inputs, trg_seqs, trg_lengths) in enumerate(data_loader_train):
            src_gcn_inputs = src_inputs['src_gcn_inputs']
            src_gcn_inputs = list(DataLoader(src_gcn_inputs, len(src_gcn_inputs)))[0].to(device)
            num_nodes_batch.append(src_gcn_inputs.num_nodes)


    initial_lr = config.lr_coverage if config.is_coverage else config.lr
    optimizer = optim.Adam(params, lr=initial_lr)
    # optimizer = optim.Adam(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

    # best_valid_loss = 1.19
    best_valid_loss = float('inf')


    # if load_model:
    #     if model_type=='gcn_lstm':
    #         model.encoder_gcn.load_state_dict(torch.load(config.save_model_path + '_encoder_gcn', map_location={'cuda:1':'cuda:3'}))
    #         model.encoder_lstm.load_state_dict(torch.load(config.save_model_path + '_encoder_lstm', map_location={'cuda:1':'cuda:3'}))
    #     else:
    #         model.encoder.load_state_dict(torch.load(config.save_model_path + '_encoder'))
    #
    #     model.decoder.load_state_dict(torch.load(config.save_model_path + '_decoder', map_location={'cuda:1':'cuda:3'}))
    #     model.reduce_state.load_state_dict(torch.load(config.save_model_path + '_reduce_state', map_location={'cuda:1':'cuda:3'}))


    if config.load_model:
        if model_type=='gcn_gtr2':
            # model.encoder_gcn.load_state_dict(torch.load(config.save_model_path + '_encoder_gcn'))
            model.encoder_gcn.load_state_dict(torch.load(config.save_model_path + '_encoder_gcn', map_location=torch.device('cpu')))
            # model.encoder_gtr.load_state_dict(torch.load(config.save_model_path + '_encoder_gtr'))
            model.encoder_gtr.load_state_dict(torch.load(config.save_model_path + '_encoder_gtr', map_location=torch.device('cpu')))
        else:
            model.encoder.load_state_dict(torch.load(config.save_model_path + '_encoder', map_location='cpu'))

        # model.decoder.load_state_dict(torch.load(config.save_model_path + '_decoder'))
        model.decoder.load_state_dict(torch.load(config.save_model_path + '_decoder', map_location=torch.device('cpu')))
        # model.reduce_state.load_state_dict(torch.load(config.save_model_path + '_reduce_state'))
        model.reduce_state.load_state_dict(torch.load(config.save_model_path + '_reduce_state', map_location=torch.device('cpu')))



    if config.mode == 'train':
        for epoch in range(config.n_epoch):
            train_loss = train(model, data_loader_train, optimizer, epoch=epoch, id2word=id2word)
            valid_loss = evaluate(model, data_loader_val)
            print('Train Loss: {} | Train PPL: {}'.format(train_loss, math.exp(train_loss)))
            print('Val.  Loss: {} | Val.  PPL: {}'.format(valid_loss, math.exp(valid_loss)))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if model_type=='gcn_gtr2':
                    torch.save(model.encoder_gcn.state_dict(), config.save_model_path+'_encoder_gcn')
                    torch.save(model.encoder_gtr.state_dict(), config.save_model_path+'_encoder_gtr')
                else:
                    torch.save(model.encoder.state_dict(), config.save_model_path + '_encoder')
                torch.save(model.decoder.state_dict(), config.save_model_path+'_decoder')
                torch.save(model.reduce_state.state_dict(), config.save_model_path+'_reduce_state')
    elif config.mode == 'translate':
        # beam_search(model, data_loader_test, word2id, id2word, rplc_dict)
        beam_search(model, data_loader_test, word2id, id2word)



if __name__ == '__main__':
    main()

#  nohup python main_PG.py > log/GCN_l2_seed43.log 2>&1 &
