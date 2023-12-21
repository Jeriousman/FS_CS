import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_head: int, embed_dim: int):
        super(SelfAttention).__init()

        self.n_head = n_head
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.n_head

        self.qkvlayer = nn.Linear(embed_dim, embed_dim*3)
        self.ffn = nn.Linear(embed_dim, embed_dim)


    def forward(self, x: torch.Tensor, future_mask=False):
        '''
        x: (batch_size, seq_len, dim) input to selfattention layer.
        multihead_inter_shape: (baych_size, seq_len, n_head, head_dim)
        
        '''
        input_shape = x.shape
        batch_size, sequence_length, embedding_size = input_shape

        ## (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) -> 3 tensors of (batch_size, seq_len, dim)
        q, k, v = self.qkvlayer(x).chunk(3, -1)

        multihead_inter_shape = (batch_size, sequence_length, self.n_head, self.head_dim)

        ## (batch_size, seq_len, dim) -> (batch_size, seq_len, n_head, head_dim) -> (batch_size, n_head, seq_len, head_dim)
        q = q.view(multihead_inter_shape).transpose(1, 2)
        k = k.view(multihead_inter_shape).transpose(1, 2)
        v = v.view(multihead_inter_shape).transpose(1, 2)

        ## (batch_size, n_head, seq_len, head_dim) @ (batch_size, n_head, head_dim, seq_len) -> (batch_size, n_head, seq_len, seq_len)
        qk_weight = q @ k.transpose(-1, -2)  

        if future_mask:
            mask = torch.ones_like(qk_weight, dtype=torch.bool).triu(diagonal = 1)  ##if diagonal =1, the diagonal lives by 1
            qk_weight.masked_fill(mask, -torch.inf)


        qk_weight /= math.sqrt(self.head_dim)
        qk_weight_softmaxed = F.softmax(qk_weight, dim=-1)

        ## (batch_size, n_head, seq_len, seq_len) @ (batch_size, n_head, seq_len, head_dim) -> (batch_size, n_head, seq_len, head_dim)
        output = qk_weight_softmaxed @ v
        
        ## (batch_size, n_head, seq_len, head_dim) -> (batch_size, seq_len, n_head, head_dim)
        output = output.transpose(1, 2)

        ## (batch_size, seq_len, dim)
        output = output.reshape(input_shape)

        ## (batch_size, seq_len, dim)
        output = self.ffn(output)

        return output




class CrossAttention(nn.Module):
    def __init__(self, n_head: int, kv_dim: int, q_dim: int):
        super(CrossAttention).__init__()

        
        # self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        self.q_dim = q_dim
        self.n_head = n_head
        assert self.q_dim // self.n_head, 'embed_dim must be divisible by n_head'

        self.qlayer = nn.Linear(q_dim, q_dim)
        self.klayer = nn.Linear(kv_dim, q_dim)
        self.vlayer = nn.Linear(kv_dim, q_dim)

        self.ffn = nn.Linear(q_dim, q_dim)

    
    def forward(self, x, y):  ## two different inputs x and y for cross attention
        '''
        x: first input  (batch_size, seq_len (h*w), q_dim). x gets query 
        y: second input (batch_size, seq_len (h*w), kv_dim. y gets key and value
        '''        
        x_inputshape = x.shape
        y_inputshape = y.shape

        batch_size, seq_len, self.q_dim = x_inputshape
        

        ## (batch_size, seq_len, q_dim) -> (batch_size, seq_len, self.n_head, self.q_dim) -> 
        q = self.qlayer(x)
        ## (batch_size, seq_len, kv_dim) -> (batch_size, seq_len, self.n_head, self.q_dim) -> 
        k = self.klayer(y)
        v = self.vlayer(y)

        multihead_inter_shape = (batch_size, seq_len, self.n_head, self.q_dim)

        ## (batch_size, seq_len, q_dim) -> (batch_size, seq_len, self.n_head, self.q_dim) -> (batch_size, self.n_head, seq_len, self.q_dim)
        q = q.view(multihead_inter_shape).transpose(1, 2)
        k = k.view(multihead_inter_shape).transpose(1, 2)
        v = v.view(multihead_inter_shape).transpose(1, 2)
           
        ## (batch_size, self.n_head, self.q_dim, seq_len) -> (batch_size, self.n_head, seq_len, seq_len) 
        qk_weight = q @ k.transpose(-1, -2)  
        qk_weight_softmaxed = F.softmax(qk_weight, dim = -1)
        ## (batch_size, self.n_head, seq_len, seq_len) @ (batch_size, self.n_head, self.q_dim, seq_len) ->  (batch_size, self.n_head, self.q_dim, seq_len)
        output = qk_weight_softmaxed @ v
        ## (batch_size, self.n_head, self.q_dim, seq_len) -> (batch_size, self.q_dim, self.n_head, seq_len)
        output = output.transpose(1, 2)

        output = output.view(x_inputshape)
        output = self.ffn(output)
        
        ## (batch_size, seq_len, q_dim)
        return output








class FlowFaceCrossAttention(nn.Module):
    def __init__(self, n_head: int, k_dim: int, q_dim: int, kv_dim: int):
        super(CrossAttention).__init__()
        '''
        Paper FLowFace uses Cross attention where values are stemming from both key and query
        '''
        
        # self.embed_dim = embed_dim
        self.k_dim = k_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.n_head = n_head
        assert self.q_dim // self.n_head, 'embed_dim must be divisible by n_head'

        self.qlayer = nn.Linear(q_dim, q_dim)
        self.qvlayer = nn.Linear(q_dim, q_dim)
        
        self.klayer = nn.Linear(kv_dim, q_dim)
        self.kvlayer = nn.Linear(kv_dim, q_dim)

        self.ffn = nn.Linear(q_dim, q_dim)

    
    def forward(self, x, y):  ## two different inputs x and y for cross attention
        '''
        x: first input  (batch_size, seq_len (h*w), q_dim). x gets query. Query is Target Face
        y: second input (batch_size, seq_len (h*w), k_dim. y gets key. Key is Source Face
        '''        
        x_inputshape = x.shape
        y_inputshape = y.shape

        batch_size, seq_len, self.q_dim = x_inputshape
        

        ## (batch_size, seq_len, q_dim) -> (batch_size, seq_len, self.n_head, self.q_dim) -> 
        q = self.qlayer(x)
        qv = self.qvlayer(x)
        ## (batch_size, seq_len, kv_dim) -> (batch_size, seq_len, self.n_head, self.q_dim) -> 
        k = self.klayer(y)
        kv = self.kvlayer(y)
        

        multihead_inter_shape = (batch_size, seq_len, self.n_head, self.q_dim)

        ## (batch_size, seq_len, q_dim) -> (batch_size, seq_len, self.n_head, self.q_dim) -> (batch_size, self.n_head, seq_len, self.q_dim)
        q = q.view(multihead_inter_shape).transpose(1, 2)
        qv = qv.view(multihead_inter_shape).transpose(1, 2)
        
        k = k.view(multihead_inter_shape).transpose(1, 2)
        kv = kv.view(multihead_inter_shape).transpose(1, 2)
        
           
        ## (batch_size, self.n_head, self.q_dim, seq_len) -> (batch_size, self.n_head, seq_len, seq_len) 
        qk_weight = q @ k.transpose(-1, -2)  
        qk_weight_softmaxed = F.softmax(qk_weight, dim = -1)
        ## (batch_size, self.n_head, seq_len, seq_len) @ (batch_size, self.n_head, self.q_dim, seq_len) ->  (batch_size, self.n_head, self.q_dim, seq_len)
        output = qk_weight_softmaxed @ kv + qv
        ## (batch_size, self.n_head, self.q_dim, seq_len) -> (batch_size, self.q_dim, self.n_head, seq_len)
        output = output.transpose(1, 2)

        output = output.view(x_inputshape)
        output = self.ffn(output)
        
        ## (batch_size, seq_len, q_dim)
        return output





class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  ##image size는 바뀌지 않고 그대로.
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)


class FeedForward(nn.module):
    def __init__(self, in_channel: int, out_channel: int):#, activation: str='relu'):
        super.__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        # self.activation = activation
        
        self.linear1 = nn.Linear(in_channel, out_channel)
        self.linear2 = nn.Linear(out_channel, out_channel)
        self.linear3 = nn.Linear(out_channel, out_channel)
        
        
        
    def forward(self, x: torch.Tensor):
        
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        
        return x
        

    


class FlowFaceCrossAttentionBlock(nn.module):
    def __init__(self, n_head: int, q_dim: int, k_dim: int, kv_dim: int):
        super().__init__()
        
        self.n_head = n_head
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.kv_dim = kv_dim
        
        self.ffcrossattention = FlowFaceCrossAttention(self.n_head, self.k_dim, self.q_dim, self.kv_dim)
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlplayer = FeedForward(self.q_dim, self.q_dim)
        self.attentionlayer1 = SelfAttention(n_head=self.n_head, embed_dim=self.q_dim)
        self.attentionlayer2 = SelfAttention(n_head=self.n_head, embed_dim=self.q_dim)



    def forward(self, target, source):
        residual_target = target
        # residual_source = source
        emb = self.ffcrossattention(target, source)
        emb = emb + residual_target  ##residual connection
        emb = self.layernorm(emb)
        residual_ln = emb
        emb = self.mlplayer(emb)
        emb = emb + residual_ln ##residual connection of embedding after layernorm and after mlp layer
        emb = self.attentionlayer1(emb)
        emb = self.attentionlayer2(emb)
        
        return emb
        