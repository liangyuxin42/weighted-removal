import pandas as pd
import torch


def load_datasets(bert_tokenizer, embedding, word_simi_train_file, word_simi_test_file, analogy_test_file, text_simi_test_file):
    # word_simi
    word_simi_train = pd.read_csv(word_simi_train_file)
    word_simi_test = pd.read_csv(word_simi_test_file)

    #save tensor in dataframe!!!
    e1_list = []
    e2_list = []
    for i in range(word_simi_train.shape[0]):
        word1 = word_simi_train['word1'][i]
        word2 = word_simi_train['word2'][i]
        id1 = torch.tensor(bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(word1)))
        id2 = torch.tensor(bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(word2)))
        e1 = torch.sum(embedding(id1).detach(), 0)
        e2 = torch.sum(embedding(id2).detach(), 0)
        e1_list.append(e1)
        e2_list.append(e2)
    word_simi_train['e1'] = e1_list
    word_simi_train['e2'] = e2_list
    print("word similarity training dataset shape:", word_simi_train.shape)


    e1_list = []
    e2_list = []
    for i in range(word_simi_test.shape[0]):
        word1 = word_simi_test['word1'][i]
        word2 = word_simi_test['word2'][i]
        id1 = torch.tensor(bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(word1)))
        id2 = torch.tensor(bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(word2)))
        e1 = torch.sum(embedding(id1).detach(), 0)
        e2 = torch.sum(embedding(id2).detach(), 0)
        e1_list.append(e1)
        e2_list.append(e2)
    word_simi_test['e1'] = e1_list
    word_simi_test['e2'] = e2_list
    print("word similarity testing dataset shape:",word_simi_test.shape)



    # anaolgy
    analogy_test = pd.read_csv(analogy_test_file)
    print(analogy_test.shape)
    print(analogy_test.head())

    #过滤
    word1_id = []
    word2_id = []
    word3_id = []
    word4_id = []

    for i in range(analogy_test.shape[0]):
        id1 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(analogy_test['word1'][i]))
        id2 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(analogy_test['word2'][i]))
        id3 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(analogy_test['word3'][i]))
        id4 = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(analogy_test['word4'][i]))
        
        word1_id.append(id1)
        word2_id.append(id2)
        word3_id.append(id3)
        word4_id.append(id4)
        
    word1_id = [word1_id[i][0] for i in range(len(word1_id))]
    word2_id = [word2_id[i][0] for i in range(len(word2_id))]
    word3_id = [word3_id[i][0] for i in range(len(word3_id))]
    word4_id = [word4_id[i][0] for i in range(len(word4_id))]
    analogy_test['id1'] = word1_id
    analogy_test['id2'] = word2_id
    analogy_test['id3'] = word3_id
    analogy_test['id4'] = word4_id

    print("word analogy dataset shape:", analogy_test.shape)

    #text similarity
    text_simi_test = pd.read_csv(text_simi_test_file)
    print("textual similarity dataset shape:", text_simi_test.shape)

    return word_simi_train, word_simi_test, analogy_test, text_simi_test

def load_datasets_forw2v(embedding, word_simi_train_file, word_simi_test_file, analogy_test_file, text_simi_test_file):
    # word_simi
    word_simi_train = pd.read_csv(word_simi_train_file)
    word_simi_test = pd.read_csv(word_simi_test_file)

    #save tensor in dataframe!!!
    e1_list = []
    e2_list = []
    for i in range(word_simi_train.shape[0]):
        id1 = word_simi_train['id1'][i]
        id2 = word_simi_train['id2'][i]
        e1 = embedding[id1]
        e2 = embedding[id2]
        e1_list.append(e1)
        e2_list.append(e2)
    word_simi_train['e1'] = e1_list
    word_simi_train['e2'] = e2_list
    print(word_simi_train.shape)
    print(word_simi_train.head())

    e1_list = []
    e2_list = []
    for i in range(word_simi_test.shape[0]):
        id1 = word_simi_test['id1'][i]
        id2 = word_simi_test['id2'][i]
        e1 = embedding[id1]
        e2 = embedding[id2]
        e1_list.append(e1)
        e2_list.append(e2)
    word_simi_test['e1'] = e1_list
    word_simi_test['e2'] = e2_list
    print(word_simi_test.shape)
    print(word_simi_test.head())


    # anaolgy
    analogy_test = pd.read_csv(analogy_test_file)
    print(analogy_test.shape)
    print(analogy_test.head())


    #text similarity
    text_simi_test = pd.read_csv(text_simi_test_file)
    print(text_simi_test.shape)
    print(text_simi_test.head())

    return word_simi_train, word_simi_test, analogy_test, text_simi_test