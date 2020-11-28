import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self,emb_dimension,component_num):
        super(BaseModel, self).__init__()
        self.emb_dimension = emb_dimension
        self.component_num = component_num
    
    def __get_cosine_similar__(self, e1, e2):
        #注意输入输出都需要是二维的，即必须以batch形式传入向量
        #e1, e2 :batch * emb_demension
        #output simi: batch
        dot = torch.bmm(e1.view(e1.shape[0],1,e1.shape[1]),e2.view(e2.shape[0],e2.shape[1],1))
        norm = torch.norm(e1,dim=1)*torch.norm(e2,dim=1)
        cosine_simi = (dot.view(dot.shape[0])/norm).double()
        return cosine_simi




class Percoefficient_Model(BaseModel):
    def __init__(self,emb_dimension,component_num,U):
        super(Percoefficient_Model, self).__init__(emb_dimension, component_num)
        self.U = U
        self.weight = nn.Parameter(torch.randn((1,self.component_num),requires_grad=True))
        

    def forward(self, x):
        # x:2*emb_demention
        # return cosine_simi预测值
        coe = torch.matmul(x, self.U)                      # (2, emb_dimension) x (emb_dimension, d) -> (2, d)
        weight = torch.Tensor.repeat(self.weight,(2,1))
        weighted_coe = torch.mul(coe, weight)              # (2, d) x (2, d) -> (2, d)
        weighted_u = torch.matmul(weighted_coe, self.U.T)  # (2, d) x (d, emb_dimension) —> (2, emb_dimension)
        weighted_e = x - weighted_u

        e1 = weighted_e[:,0]
        e2 = weighted_e[:,1]
        cosine_simi = self.__get_cosine_similar__(e1, e2) 
        #cosine_simi = torch.div(torch.add(cosine_simi,1.0),2.0) #[-1~1]->[0~1]
        
        return cosine_simi #返回最终结果