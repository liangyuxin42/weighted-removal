# encoding:utf-8
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import tools.testForTask as testForTask
import tools.draw as draw

def visualize(overall):
    # visualize overall results
    # word similarity
    task_name = 'word_simi'
    original_result = overall[task_name][0]
    cn_result = overall[task_name][1]
    abtt_result_ds = overall[task_name][2:2+len(D)]
    wr_result_ds = overall[task_name][2+len(D):]

    draw.draw_overall(original_result, cn_result, abtt_result_ds, wr_result_ds,D,task_name,emb_type)

    # word analogy
    task_name = 'analogy'
    original_result = overall[task_name][0]
    cn_result = overall[task_name][1]
    abtt_result_ds = overall[task_name][2:2+len(D)]
    wr_result_ds = overall[task_name][2+len(D):]

    draw.draw_overall(original_result, cn_result, abtt_result_ds, wr_result_ds,D,task_name,emb_type)

    # textual similarity
    task_name = 'text_simi'
    original_result = overall[task_name][0]
    cn_result = overall[task_name][1]
    abtt_result_ds = overall[task_name][2:2+len(D)]
    wr_result_ds = overall[task_name][2+len(D):]

    draw.draw_overall(original_result, cn_result, abtt_result_ds, wr_result_ds,D,task_name,emb_type)


def evaluate(args):

    model_name = args.bert_model
    if model_name == 'bert-base-uncased':
        emb_type = 'base'
    if model_name == 'bert-large-uncased':
        emb_type = 'large'
    # all dataset path for training
    word_simi_train_file = config['word_simi_train_file']
    word_simi_test_file = config['word_simi_test_file']
    analogy_test_file = config['analogy_test_file']
    text_simi_test_file = config['text_simi_test_file']
    # list ds for testing
    D = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]

    bert_model = BertModel.from_pretrained(model_name)
    bert_model.eval()
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    embedding = bert_model.get_input_embeddings()

    ids = torch.tensor(range(30522))
    E = embedding(ids).detach().numpy()
    print('BERT Embedding shape check:', E.shape)

    emb_dimension = E.shape[1]
    vocab_len = E.shape[0]
    #load U
    U = np.load('trained-embedding//U_%s.npy' % emb_type )
    E = torch.tensor(E)
    word_simi_train, word_simi_test,analogy_test, text_simi_test = \
    loaddatasets.load_datasets(bert_tokenizer, embedding, word_simi_train_file, word_simi_test_file, analogy_test_file, text_simi_test_file)

    # add results of original embedding to dataframe
    print('evaluating on the original embedding...')
    word_simi_result = testForTask.word_similarity(word_simi_test,E.numpy(),bert_tokenizer,0,'original')
    text_simi_result = testForTask.text_similarity(text_simi_test, E.numpy(), bert_tokenizer,0,'original')
    analogy_result = testForTask.word_analogy(analogy_test,E.numpy(), bert_tokenizer,0,'original')

    #get embedding processed by CN
    x_collector = E.numpy().T
    #x_collector.shape = (emb_dim, V)
    nrWords = x_collector.shape[1] # number of total words
    R = x_collector.dot(x_collector.T) / nrWords # calculate the un-centered correlation matrix
    #R.shape = (emb_dim, emb_dim)
    C = R @ np.linalg.inv(R + 2 ** (-2) * np.eye(emb_dimension))# calculate the conceptor matrix
    #C.shape = (emb_dim, emb_dim)
    vecMatrix = ((np.eye(emb_dimension) - C) @ x_collector).T 
    #vecMatrix.shape = (V, emb_dim) -> NEW EMB
    CN_emb = vecMatrix

    # add results of CN embedding to dataframe
    print('evaluating CN method...')
    word_simi_result = testForTask.word_similarity(word_simi_test,CN_emb,bert_tokenizer,0,'cn')
    text_simi_result = testForTask.text_similarity(text_simi_test, CN_emb, bert_tokenizer,0,'cn')
    analogy_result = testForTask.word_analogy(analogy_test,CN_emb, bert_tokenizer,0,'cn')

    #calculate results for each abtt-d embedding and add to dataframe
    print('evaluating ABTT method...')
    for d in D:
        print("d = ", d)
        
        #get abtt-d embedding
        coe = torch.matmul(E,torch.tensor(U[:d]).T)
        U_coe = torch.matmul(coe, torch.tensor(U[:d]))
        ABTT_Emb = E - U_coe
        
        #calculate results
        word_simi_result = testForTask.word_similarity(word_simi_test,ABTT_Emb.numpy(),bert_tokenizer,d,'abtt')
        text_simi_result = testForTask.text_similarity(text_simi_test, ABTT_Emb.numpy(), bert_tokenizer,d,'abtt')
        analogy_result = testForTask.word_analogy(analogy_test,ABTT_Emb.numpy(), bert_tokenizer,d,'abtt')
    
    #calculate results for each wr-d embedding and add to dataframe
    print('evaluating WR method...')
    for d in D:
        print("d = ", d)
        
        WR_Emb = np.load('trained-embedding/%sEmb_%s.npy' % (emb_type, d))
        
        word_simi_result = testForTask.word_similarity(word_simi_test,WR_Emb,bert_tokenizer,d,'wr')
        text_simi_result = testForTask.text_similarity(text_simi_test, WR_Emb, bert_tokenizer,d,'wr')
        analogy_result = testForTask.word_analogy(analogy_test,WR_Emb, bert_tokenizer,d,'wr')

    word_simi_result = word_simi_result.drop(columns = ['index','id1','id2','id_num','e1','e2'])
    text_simi_result = text_simi_result.drop(columns = ['genre','filename','index'])
    analogy_result = analogy_result.drop(columns = ['index','id1','id2','id3','id4','id_num'])
    
    method_list = list(word_simi_result.columns[4:])
    overall = pd.DataFrame()
    overall['method'] = method_list
    corr_ws = []
    ana_acc = []
    corr_ts = []

    for method in method_list:
        column_name = '%s' % method
    
        corr_ws.append(word_simi_result[['simi',column_name]].corr('pearson')['simi'][column_name])
        corr_ts.append(text_simi_result[['score',column_name]].corr('pearson')['score'][column_name])
        ana_acc.append(analogy_result[column_name].sum()/analogy_result.shape[0])

    overall['word_simi'] = corr_ws
    overall['analogy'] = ana_acc
    overall['text_simi'] = corr_ts

    visualize(overall)
    overall.to_csv('results/overall.csv')

    #per dataset results
    # word similarity
    word_simi_datasets = pd.DataFrame()
    word_simi_datasets['method'] = method_list
    for name, group in word_simi_result.groupby('dataset_name'):
        corr_d = []
        for method in method_list:
            column_name = '%s' % method
            corr_pearson = group[['simi',column_name]].corr('pearson')['simi'][column_name]
            corr_d.append(corr_pearson)
        word_simi_datasets[name] = corr_d
    word_simi_datasets.to_csv('results/word_simi_ofdatasets_%s.csv' %(emb_type))

    #textual similarity
    text_simi_datasets = pd.DataFrame()
    text_simi_datasets['method'] = method_list
    for name, group in text_simi_result.groupby('year'):
        corr_d = []
        for method in method_list:
            column_name = '%s' % method
            corr_pearson = group[['score',column_name]].corr('pearson')['score'][column_name]
            corr_d.append(corr_pearson)
        text_simi_datasets[name] = corr_d
    text_simi_datasets.to_csv('results/text_simi_result_datasets_%s.csv' %(emb_type))

    #analogy:semantic vs syntactic
    analogy_parts = pd.DataFrame()
    analogy_parts['method'] = method_list
    for name, group in analogy_result.groupby('part'):
        acc = []
        for method in method_list:
            column_name = '%s' % method
            sum_d = group[column_name].sum()
            acc.append(sum_d/group.shape[0])
        analogy_parts[name] = acc
    analogy_parts.to_csv('results/analogy_part_%s.csv' %(emb_type))

    #analogy
    analogy_datasets = pd.DataFrame()
    analogy_datasets['method'] = method_list
    for name, group in analogy_result.groupby('dataset_name'):
        acc = []
        for method in method_list:
            column_name = '%s' % method
            sum_d = group[column_name].sum()
            acc.append(sum_d/group.shape[0])
        analogy_datasets[name] = acc
    analogy_datasets.to_csv('results/analogy_datasets_%s.csv' %(emb_type))

    print('all results saved')

def main():
    parser = argparse.ArgumentParser(description='WR training')

    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='choose bert model(bert-base-uncased/bert-large-uncased)')
    
    args = parser.parse_args()
    
    evaluate(args)

if __name__ == '__main__':
    main()
