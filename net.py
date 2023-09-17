import torch
import torch.nn as nn
import torch.nn.functional as F

class DSAN(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN, self).__init__()
        self.n_outputs = 288
        self.feature_layers = DCRN_02(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)

        self.head1 = nn.Sequential(
            nn.Linear(288, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(288, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features,x1,x2,fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out

class DSAN1(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN1, self).__init__()
        self.n_outputs = 288
        self.feature_layers = DCRN_02(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)

        self.head1 = nn.Sequential(
            nn.Linear(288, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(288, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features,x1,x2,fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out

class DSAN2(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN1, self).__init__()
        self.n_outputs = 152
        self.feature_layers = DCRN(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(self.n_outputs, num_class)
        self.fc2 = nn.Linear(self.n_outputs, 1)

        self.head1 = nn.Sequential(
            nn.Linear(self.n_outputs, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(self.n_outputs, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features,x1,x2,fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out

class DCRN_02(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN_02, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)
        # Finish

        # Combination shape
        # self.inter_size = 128 + 24
        self.inter_size = 192 + 96


        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                bias=True)  # padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.unsqueeze(1)  # (64,1,100,9,9)
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1  # (32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)  # (32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))  # (32,128,7,7)

        x2 = self.conv5(x)  # (32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)  # (32,24,1,7,7)
        x2 = self.conv6(x2)  # (32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)  # (32,24,1,7,7)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))  # (32,24,7,7)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)  # (32,152,7,7)

        ###################
        # attention map
        ###################
        ###################
        # attention map
        ###################
        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # (288)

        #####################
        # attention map over
        #####################
        # CMMD

        return x

class DCRN(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)#padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),bias=True)# padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24



        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),bias=True)#padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.ca = ChannelAttention(self.inter_size)#self.inter_size
        self.sa = SpatialAttention()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m,nn.Linear):
            #     torch.nn.init.kaiming_normal_(m.weight.data)
            #     m.bias.data = torch.ones(m.bias.data.size())



    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:

            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data = torch.ones(m.bias.data.size())

    def forward(self, x):

        x = x.unsqueeze(1) # (64,1,100,9,9)
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1                  #(32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)                 #(32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4)) #(32,128,7,7)

        ###########
        # attention model
        #BAM
        ###########
        # x1 = self.ca(x1) * x1


        ###########################
        #spatial

        x2 = self.conv5(x)                      #(32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)     #(32,24,1,7,7)
        x2 = self.conv6(x2)                 #(32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)                 #(32,24,1,7,7)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4)) #(32,24,7,7)

        ################
        #attention model
        ################
        # x2 = self.sa(x2) * x2
        ##SAM


        # concat spatial and spectral information
        # x1 = x1 * self.ca(x1)
        # x2 = x2 * self.sa(x2)

        x = torch.cat((x1, x2), 1)      #(32,152,7,7)


        ###################
        # attention map
        ###################
        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # (288)



        #####################
        # attention map over
        #####################
        #CMMD


        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False) #4-->16
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)