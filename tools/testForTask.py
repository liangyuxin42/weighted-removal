import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def word_similarity(df,Emb,tokenizer,d,method):

    #df中应该包括word1, word2, simi三列
    #Emb提供从id到emb_vec的方法
    
    name = '%s_%s' %(method,d)

    cosine_list = []

    for i in range(df.shape[0]):
        word1 = df['word1'][i]
        word2 = df['word2'][i]
        id1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word1))
        id2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word2))
        e1 = np.sum(Emb[id1], 0)
        e2 = np.sum(Emb[id2], 0)
        cosine_list.append(cosine_similarity([e1],[e2]))

    df[name] = np.array(cosine_list).flatten()
    
    return df


def sent2vec_byadd(sent, Emb, tokenizer):
    
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
    
    embs = np.sum(Emb[ids], 0)
    emb_avg = embs/len(ids)

    return emb_avg

def text_similarity(df, Emb, tokenizer,d,method):
    name = '%s_%s' %(method,d)

    cosine_simis = []
    for index,row in df.iterrows():
        e1 = sent2vec_byadd(row['sent1'], Emb, tokenizer)
        e2 = sent2vec_byadd(row['sent2'], Emb, tokenizer)
        cosine_simis.append(cosine_similarity([e1],[e2]))
        
    df[name] = np.array(cosine_simis).flatten()


    return df

def word_analogy(df,all_Emb, tokenizer,d,method):
    # 需要预先处理ID
    name = '%s_%s' %(method,d)

    small_number = -100

    all_norm = np.linalg.norm(all_Emb, axis=1)
    count = []

    id1_all = df['id1'].to_numpy()
    id2_all = df['id2'].to_numpy()
    id3_all = df['id3'].to_numpy()
    id4_all = df['id4'].to_numpy()

    emb1 = all_Emb[id1_all]
    emb2 = all_Emb[id2_all]
    emb3 = all_Emb[id3_all]

    emb_predict = emb3 + emb2 - emb1

    for i in range(emb_predict.shape[0]):
        #找到最相似
        emb_p = emb_predict[i]
        dot = np.dot(all_Emb,emb_p)
        norm_predict = np.linalg.norm(emb_p)
        norm = np.dot(all_norm,norm_predict)
        cosine = dot/norm
        cosine[[id1_all[i],id2_all[i],id3_all[i]]] = small_number
        id4_predict = np.argmax(cosine,axis = 0)
        if id4_predict == id4_all[i]:
            count.append(1)
        else:
            count.append(0)
    
    df[name] = count
    
    return df