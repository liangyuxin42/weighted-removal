# encoding:utf-8
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import Adam
from torch.autograd import Variable

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import *

from config_file import config

import sys
sys.path.append("..")
import tools.models as models
import tools.dataloaders as dataloaders
import tools.all_test_forBERT as all_test_forBERT
import tools.loaddatasets as loaddatasets

def train_epoch(model, data_loader, loss_fn, optimizer,device):
    
    model = model.train()
    losses = []
    
    for step,d in enumerate(data_loader):
        emb = d['emb'].to(device)
        simi_label = d['simi_label'].to(device)
        
        simi_predict = model(x = emb)
        
        loss = loss_fn(simi_predict, simi_label)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
    return losses



def train(args):
    #choose bert model
    model_name = args.bert_model
    random_state = args.random_state
    batch_size = args.batch_size
    lr = args.lr
    EPOCHS = args.EPOCHS
    # all dataset path for training
    word_simi_train_file = config['word_simi_train_file']
    word_simi_test_file = config['word_simi_test_file']
    analogy_test_file = config['analogy_test_file']
    text_simi_test_file = config['text_simi_test_file']
    #list the ds you want to train
    D = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_name == 'bert-base-uncased':
        emb_type = 'base'
    if model_name == 'bert-large-uncased':
        emb_type = 'large'

    bert_model = BertModel.from_pretrained(model_name)
    bert_model.eval()
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    embedding = bert_model.get_input_embeddings()

    ids = torch.tensor(range(30522))
    E = embedding(ids).detach().numpy()
    print('BERT Embedding shape check:', E.shape)

    emb_dimension = E.shape[1]
    vocab_len = E.shape[0]

    pca = PCA(random_state = random_state).fit(E)

        # U
    E = torch.tensor(E)
    U = pca.components_
    np.save('trained-embedding//U_%s.npy' % emb_type , U)
    U = torch.tensor(U)

    word_simi_train, word_simi_test,analogy_test, text_simi_test = \
    loaddatasets.load_datasets(bert_tokenizer, embedding, word_simi_train_file, word_simi_test_file, analogy_test_file, text_simi_test_file)
    train_loader = dataloaders.create_data_loader_forBERT(word_simi_train, batch_size, True, dataloaders.Dataset_direct2emb)
    test_loader = dataloaders.create_data_loader_forBERT(word_simi_test, batch_size, False, dataloaders.Dataset_direct2emb)

    # training
    for d in D:
        print(f'D: {d}')
        print('~' * 10)
        u = U[:d]
        u = Variable(torch.tensor(u.T), requires_grad=False).to(device)

        model = models.Percoefficient_Model(emb_dimension = emb_dimension, component_num = d, U = u).to(device)
        optimizer = Adam(model.parameters(), lr = lr)
        total_steps = len(train_loader) * EPOCHS
        loss_fn = nn.MSELoss().to(device)
        
        for epoch in range(EPOCHS):
            #标出每个EPOCHS的头部
            print('-' * 10)
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            
            train_loss = train_epoch(
            model,
            train_loader,
            loss_fn, 
            optimizer,
            device
        )
            epoch_loss = np.mean(train_loss)
            print(f'Train loss {epoch_loss} ')
        
        x = []
        for parameters in model.parameters():
            print(parameters)
            x.append(parameters)
        para = x[0].sum(axis = 0).cpu().detach()

        u_cpu = u.cpu().detach()
        coe = torch.matmul(E,u_cpu)
        weighted_coe = torch.mul(para,coe)
        weighted_u = torch.matmul(weighted_coe,u_cpu.T)
        
        Emb = (E-weighted_u).numpy()
        np.save('trained-embedding/%sEmb_%s.npy' %(emb_type, d),Emb)
        torch.save(model,'trained-model/%s_%s_%s.pth' %(emb_type, d, EPOCHS))
        print('%s_%s_%s model saved' %(emb_type, d, EPOCHS) )
    
    print('training finish!')


def main():
    parser = argparse.ArgumentParser(description='WR training')

    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='choose bert model(bert-base-uncased/bert-large-uncased)')
    parser.add_argument('--random_state', default=42, type=int,
                        help='random_state')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='learning rate')
    parser.add_argument('--EPOCHS', default=200, type=int,
                        help='epoch for training')
    args = parser.parse_args()
    
    train(args)

if __name__ == '__main__':
    main()