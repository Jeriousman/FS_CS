import torch
import torch.nn as nn
import torch.nn.functional as F



def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def conv4x4(in_c, out_c, norm=nn.BatchNorm2d, same_size=False):
    if same_size==True:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False), ## halving feature map size. e.g., 256 -> 128
            norm(out_c),
            nn.LeakyReLU(0.1)
        )
            
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False), ## halving feature map size. e.g., 256 -> 128
            norm(out_c),
            nn.LeakyReLU(0.1)
        )

class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False) ##H,W를 두개로 크게만듬
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1)
        
        self.deconv_same = nn.ConvTranspose2d(in_channels=out_c*2, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False) ####H,W를 같게 아웃푸팅함
        

    def forward(self, input, skip_tensor, backbone='unet'):
        # print('input size: ', input.shape)
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        # print('????', x.shape)
        if backbone == 'linknet':
            return x+skip_tensor
        else:
            x = torch.cat((x, skip_tensor), dim=1)
            # print('concat size: ', x.shape)
            x = self.deconv_same(x)
            # print('deconv last: ', x.shape)
            return x


class CAdeconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        '''
        Cross Attention deconvolution layer
        dimension transition: in_channel -> in_channel -> in_channel*2 -> out_channel
        '''
        super(CAdeconv4x4, self).__init__()
        self.deconv_input = nn.ConvTranspose2d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = norm(in_c)
        self.lrelu = nn.LeakyReLU(0.1)
        
        self.deconv_same = nn.ConvTranspose2d(in_channels=in_c*2, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)
        

    def forward(self, input, skip_tensor, backbone='unet'):
        # print('input: ', input.shape)
        x = self.deconv_input(input)
        # print('deconv input: ', x.shape)
        x = self.bn(x)
        x = self.lrelu(x)
        if backbone == 'linknet':
            return x+skip_tensor
        else:
            x = torch.cat((x, skip_tensor), dim=1)
            # print('deconv output after concat: ', x.shape)
            x = self.deconv_same(x)
            # print('deconv output dimension reduction: ', x.shape)
            return x



class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)



class noskip_deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(noskip_deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, input):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return x
        


class MLAttrEncoder(nn.Module):  ##Multi-level Attributes Encoder
    def __init__(self, backbone):
        super(MLAttrEncoder, self).__init__()
        self.backbone = backbone
        self.conv1 = conv4x4(3, 32)  ##256 -> 128
        self.conv2 = conv4x4(32, 64)  ## 128 -> 64
        self.conv3 = conv4x4(64, 128) ## 64 -> 32
        self.conv4 = conv4x4(128, 256) ## 32 -> 16
        self.conv5 = conv4x4(256, 512) ## 16 -> 4
        self.conv6 = conv4x4(512, 1024) ## 4 -> 2
        self.conv7 = conv4x4(1024, 1024) ## 2 - > 1
        
        if backbone == 'unet':
            self.deconv1 = deconv4x4(1024, 1024) 
            self.deconv2 = deconv4x4(2048, 512)
            self.deconv3 = deconv4x4(1024, 256)
            self.deconv4 = deconv4x4(512, 128)
            self.deconv5 = deconv4x4(256, 64)
            self.deconv6 = deconv4x4(128, 32)
        elif backbone == 'linknet':
            self.deconv1 = deconv4x4(1024, 1024)
            self.deconv2 = deconv4x4(1024, 512)
            self.deconv3 = deconv4x4(512, 256)
            self.deconv4 = deconv4x4(256, 128)
            self.deconv5 = deconv4x4(128, 64)
            self.deconv6 = deconv4x4(64, 32)
        self.apply(weight_init)

    def forward(self, Xt):
        feat1 = self.conv1(Xt)
        # 32x128x128
        feat2 = self.conv2(feat1)
        # 64x64x64
        feat3 = self.conv3(feat2)
        # 128x32x32
        feat4 = self.conv4(feat3)
        # 256x16xx16
        feat5 = self.conv5(feat4)
        # 512x8x8
        feat6 = self.conv6(feat5)
        # 1024x4x4
        z_attr1 = self.conv7(feat6)
        # 1024x2x2

        z_attr2 = self.deconv1(z_attr1, feat6, self.backbone)
        z_attr3 = self.deconv2(z_attr2, feat5, self.backbone)
        z_attr4 = self.deconv3(z_attr3, feat4, self.backbone)
        z_attr5 = self.deconv4(z_attr4, feat3, self.backbone)
        z_attr6 = self.deconv5(z_attr5, feat2, self.backbone)
        z_attr7 = self.deconv6(z_attr6, feat1, self.backbone)
        z_attr8 = F.interpolate(z_attr7, scale_factor=2, mode='bilinear', align_corners=True)
        return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8

    
class AADGenerator(nn.Module):
    def __init__(self, backbone, c_id=256, num_blocks=2):
        super(AADGenerator, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AAD_ResBlk(1024, 1024, 1024, c_id, num_blocks)
        if backbone == 'linknet':
            self.AADBlk2 = AAD_ResBlk(1024, 1024, 1024, c_id, num_blocks)
            self.AADBlk3 = AAD_ResBlk(1024, 1024, 512, c_id, num_blocks)
            self.AADBlk4 = AAD_ResBlk(1024, 512, 256, c_id, num_blocks)
            self.AADBlk5 = AAD_ResBlk(512, 256, 128, c_id, num_blocks)
            self.AADBlk6 = AAD_ResBlk(256, 128, 64, c_id, num_blocks)
            self.AADBlk7 = AAD_ResBlk(128, 64, 32, c_id, num_blocks)
            self.AADBlk8 = AAD_ResBlk(64, 3, 32, c_id, num_blocks)
        else:
            self.AADBlk2 = AAD_ResBlk(1024, 1024, 2048, c_id, num_blocks)
            self.AADBlk3 = AAD_ResBlk(1024, 1024, 1024, c_id, num_blocks)
            self.AADBlk4 = AAD_ResBlk(1024, 512, 512, c_id, num_blocks)
            self.AADBlk5 = AAD_ResBlk(512, 256, 256, c_id, num_blocks)
            self.AADBlk6 = AAD_ResBlk(256, 128, 128, c_id, num_blocks)
            self.AADBlk7 = AAD_ResBlk(128, 64, 64, c_id, num_blocks)
            self.AADBlk8 = AAD_ResBlk(64, 3, 64, c_id, num_blocks)
        
        self.apply(weight_init)

    def forward(self, z_attr, z_id):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        m2 = F.interpolate(self.AADBlk1(m, z_attr[0], z_id), scale_factor=2, mode='bilinear', align_corners=True) ##F.interpolate 작은 사이즈의 이미지를 큰 사이즈로 키울 때 사용된다
        m3 = F.interpolate(self.AADBlk2(m2, z_attr[1], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m4 = F.interpolate(self.AADBlk3(m3, z_attr[2], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m5 = F.interpolate(self.AADBlk4(m4, z_attr[3], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m6 = F.interpolate(self.AADBlk5(m5, z_attr[4], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m7 = F.interpolate(self.AADBlk6(m6, z_attr[5], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m8 = F.interpolate(self.AADBlk7(m7, z_attr[6], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        y = self.AADBlk8(m8, z_attr[7], z_id)
        return torch.tanh(y)


class AEI_Net(nn.Module):
    def __init__(self, backbone, num_blocks=2, c_id=256):
        super(AEI_Net, self).__init__()
        if backbone in ['unet', 'linknet']:
            self.encoder = MLAttrEncoder(backbone)
        elif backbone == 'resnet':
            self.encoder = MLAttrEncoderResnet()
        self.generator = AADGenerator(backbone, c_id, num_blocks)

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr

    def get_attr(self, X):
        return self.encoder(X)



####################################################################################
## CrossU

class UNet(nn.Module):  ##Multi-level Attributes Encoder
    def __init__(self, backbone):
        super(UNet, self).__init__()
        self.backbone = backbone
        self.conv1 = conv4x4(3, 32, same_size=True)  ##256 -> 32x256x256
        self.conv2 = conv4x4(32, 64)  ##32x256x256 -> 64x128x128
        self.conv3 = conv4x4(64, 128)  ## 64x128x128 -> 128x64x64
        self.conv4 = conv4x4(128, 256) ## 128x64x64 -> 256x32x32
        self.conv5 = conv4x4(256, 512) ## 256x32x32 -> 512x16x16
        self.conv6 = conv4x4(512, 1024) ## 512x16x16 -> 1024x8x8
        self.conv7 = conv4x4(1024, 1024) ## 1024x8x8 -> 1024x4x4  ##bottleneck
          
        if backbone == 'unet':
            self.deconv1 = deconv4x4(1024, 1024)  ## 4 ->  8  (1024x8x8)
            self.deconv2 = deconv4x4(1024, 512)  ## 8 ->  16 (512x16x16)
            self.deconv3 = deconv4x4(512, 256)  ## 16 > 32  (256x32x32)
            self.deconv4 = deconv4x4(256, 128) ## 32 > 64 (128x64x64)
            self.deconv5 = deconv4x4(128, 64) ## 64 > 128  (64x128x128)
            self.deconv6 = deconv4x4(64, 32) ## 128 > 256  (32x256x256)
            self.deconv7 = outconv(32, 3) ## 256 > 256  (3x256x256)  
            
        elif backbone == 'linknet':
            self.deconv1 = deconv4x4(1024, 1024)
            self.deconv2 = deconv4x4(1024, 512)
            self.deconv3 = deconv4x4(512, 256)
            self.deconv4 = deconv4x4(256, 128)
            self.deconv5 = deconv4x4(128, 64)
            self.deconv6 = deconv4x4(64, 32)
        self.apply(weight_init)

    def forward(self, x):
        ##256 -> 32x256x256
        feat1 = self.conv1(x) 
        # 64x128x128
        feat2 = self.conv2(feat1)
        # 128x64x64
        feat3 = self.conv3(feat2)
        # 256x32x32
        feat4 = self.conv4(feat3)
        # 512x16x16
        feat5 = self.conv5(feat4)
        # 1024x8xx8
        feat6 = self.conv6(feat5)
        
        # 1024x4x4  the bottlebeck    
        bottlneck_attr = self.conv7(feat6)
        
        
        ## 1024x4x4 -> 1024x8x8 = z_attr1
        z_attr1 = self.deconv1(bottlneck_attr, feat6, self.backbone) ## 4 > 8  ##feat4 = skip connection the same size with same level counterpart
        ## 1024x8x8 -> 512x16x16
        z_attr2 = self.deconv2(z_attr1, feat5, self.backbone) ## 8 > 16 
        ## 512x16x16 -> 256x32x32
        z_attr3 = self.deconv3(z_attr2, feat4, self.backbone) ## 16 > 32
        ## 256x32x32 -> 128x64x64
        z_attr4 = self.deconv4(z_attr3, feat3, self.backbone) ## 32 > 64
        ## 128x64x64 -> 64x128x128
        z_attr5 = self.deconv5(z_attr4, feat2, self.backbone) ## 64 > 128
        ## 64x128x128 -> 32x256x256
        z_attr6 = self.deconv6(z_attr5, feat1, self.backbone) ## 128 > 256
        ## 32x256x256 -> 3x256x256
        z_attr7 = self.deconv7(z_attr6) ## 128 > 256
        
        
        return bottlneck_attr, z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7