import torch
from torch.utils.data import Dataset, DataLoader,random_split


#dataloader
class Dataset_id2hot(Dataset):
    def __init__(self, id1, id2, simi_label, E):
        #dataloading
        self.id1 = id1
        self.id2 = id2
        self.label = simi_label
        self.E = E
        
    def __getitem__(self,index):
        #dataset[index]
        word2hot_vec = np.zeros([2,vocab_len])
        word2hot_vec[0][self.id1[index]] = 1
        word2hot_vec[1][self.id2[index]] = 1
        word2hot_vec = torch.tensor(word2hot_vec)
        
        simi_label = torch.tensor(self.label[index]).double()
        
        return {
            'word2hot' : word2hot_vec,
            'simi_label' : simi_label
            }
            
    def __len__(self):
        #len(dataset)
        return len(self.id1)


class Dataset_id2emb(Dataset):
    def __init__(self, id1, id2, simi_label, E):
        #dataloading
        self.id1 = id1
        self.id2 = id2
        self.label = simi_label
        self.E = E
        
    def __getitem__(self,index):
        #dataset[index]
        e1 = torch.tensor(self.E[self.id1[index]])
        e2 = torch.tensor(self.E[self.id2[index]])
        
        emb = torch.cat((e1.view(1,e1.shape[0]), e2.view(1,e2.shape[0])), 0)
        #print(emb.shape)
        simi_label = torch.tensor(self.label[index]).double()
        
        return {
            'emb' : emb,
            'simi_label' : simi_label
            }
            
    def __len__(self):
        #len(dataset)
        return len(self.id1)

class Dataset_word2emb(Dataset):
    # for bert so tokenizer is needed
    def __init__(self, word1, word2, simi_label, E, tokenizer):
        #dataloading
        self.word1 = word1
        self.word2 = word2
        self.tokenizer = tokenizer
        self.E = E
        self.label = simi_label
           
        
    def __getitem__(self,index):
        #dataset[index]
        word1 = self.word1[index]
        word2 = self.word2[index]
        id1 = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word1)))
        id2 = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word2)))
        e1 = torch.sum(self.E(id1).detach(), 0)
        e2 = torch.sum(self.E(id2).detach(), 0)

        emb = torch.cat((e1.view(1,e1.shape[0]), e2.view(1,e2.shape[0])), 0)
        #print(emb.shape)
        simi_label = torch.tensor(self.label[index]).double()
        
        return {
            'emb' : emb,
            'simi_label' : simi_label
            }
            
    def __len__(self):
        #len(dataset)
        return len(self.word1)

class Dataset_direct2emb(Dataset):
    # for bert so tokenizer is needed
    def __init__(self,emb1, emb2, simi_label):
        #dataloading
        self.emb1 = emb1
        self.emb2 = emb2
        self.label = simi_label
           
        
    def __getitem__(self,index):
        #dataset[index]
        e1 = torch.tensor(self.emb1[index])
        e2 = torch.tensor(self.emb2[index])

        emb = torch.cat((e1.view(1,e1.shape[0]), e2.view(1,e2.shape[0])), 0)
        #print(emb.shape)
        simi_label = torch.tensor(self.label[index]).double()
        
        return {
            'emb' : emb,
            'simi_label' : simi_label
            }
            
    def __len__(self):
        #len(dataset)
        return len(self.label)

def create_data_loader(df, batch_size, shuffle, dataset,E):

    ds = dataset(
        id1 = df['id1'].to_numpy(),
        id2 = df['id2'].to_numpy(),
        simi_label = df['simi'].to_numpy(),
        E = E.numpy()
    )

    return DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = shuffle
      )

def create_data_loader_forBERT(df, batch_size, shuffle, dataset):

    ds = dataset(
        emb1 = df['e1'],
        emb2 = df['e2'],
        simi_label = df['simi'].to_numpy()
    )

    return DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = shuffle
      )