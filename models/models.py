import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from .config import device, num_classes


def create_model(opt):
    if opt.model == 'pix2pixHD':
        #from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        from .fs_model import fsModel
        model = fsModel()
    else:
        from .ui_model import UIModel
        model = UIModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)

        return x


class ArcMarginModel(nn.Module):
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output



class SelfAttentionLayer(nn.Module):
    def __init__(self, n_head: int, embed_dim: int, cross_embed_dim: int):
        super(SelfAttentionLayer, self).__init()
        '''
        embed_dim = 전체 엠베딩 디멘션 (n_head로 나누기 전의 엠베딩 디멘션)
        '''
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.n_head


        self.qlayer = nn.Linear(embed_dim, embed_dim)
        self.klayer = nn.Linear(cross_embed_dim, embed_dim)
        self.vlayer = nn.Linear(cross_embed_dim, embed_dim)
    
        self.ffn = nn.Linear(embed_dim, embed_dim)

        # self.qkvlayer = nn.Linear(embed_dim, embed_dim*3)
        # self.ffn = nn.Linear(embed_dim, embed_dim)



    def forward(self, x: torch.Tensor, future_mask=False):
        '''
        x: (batch_size, seq_len, dim) input to selfattention layer.
        multihead_inter_shape: (baych_size, seq_len, n_head, head_dim)
        
        '''
        input_shape = x.shape
        batch_size, sequence_length, embedding_size = input_shape

        ## (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) -> 3 tensors of (batch_size, seq_len, dim)
        # q, k, v = self.qkvlayer(x).chunk(3, -1)

        multihead_inter_shape = (batch_size, -1, self.n_head, self.head_dim)

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




class FlowFaceCrossAttentionLayer(nn.Module):
    def __init__(self, n_head: int, k_dim: int, q_dim: int, kv_dim: int):
        super(FlowFaceCrossAttentionLayer, self).__init__()
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

class FeedForward(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):#, activation: str='relu'):
        super(FeedForward, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.activation = activation
        
        self.linear1 = nn.Linear(self.in_dim, self.out_dim)
        self.linear2 = nn.Linear(self.in_dim, self.out_dim)
        self.linear3 = nn.Linear(self.in_dim, self.out_dim)
        
        
        
    def forward(self, x: torch.Tensor):
        
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        
        return x



class FlowFaceCrossAttentionBlock(nn.Module):
    def __init__(self, seq_len: int, n_head: int, k_dim: int, q_dim: int, kv_dim: int):
        super(FlowFaceCrossAttentionBlock, self).__init__()
        
        self.seq_len = seq_len
        self.n_head = n_head
        self.k_dim = k_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        
        
        self.FFCA = FlowFaceCrossAttentionLayer(n_head=self.n_head, k_dim=self.k_dim, q_dim=self.q_dim, kv_dim=self.kv_dim)
        self.LN = torch.nn.LayerNorm([self.seq_len, self.q_dim]) 
        self.FFN = FeedForward(in_dim=self.q_dim, out_dim=self.q_dim)
        self.SA1 = SelfAttentionLayer(n_head=self.n_head, embed_dim=self.q_dim) ##transformer (?)
        self.SA2 = SelfAttentionLayer(n_head=self.n_head, embed_dim=self.q_dim) ##transformer (?)
        
        
        
        
    def forward(self, x, y):
        
        raw_x = x
        x = self.FFCA(x)
        x = x + raw_x
        x = self.LN(x)
        inter_x = self.FFN(x)
        x = x + inter_x
        x = self.SA1(x)
        x = self.SA2(x)
        
        return x