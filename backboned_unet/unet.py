import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F


def get_backbone(name, pretrained=True):

    """ Loading backbone, defining names for skip-connections and encoder output. """

    # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        backbone = models.resnet152(pretrained=pretrained)
    elif name == 'vgg16':
        backbone = models.vgg16_bn(pretrained=pretrained).features
    elif name == 'vgg19':
        backbone = models.vgg19_bn(pretrained=pretrained).features
    elif name == 'densenet121':
        backbone = models.densenet121(pretrained=True).features
    elif name == 'densenet161':
        backbone = models.densenet161(pretrained=True).features
    elif name == 'densenet169':
        backbone = models.densenet169(pretrained=True).features
    elif name == 'densenet201':
        backbone = models.densenet201(pretrained=True).features
    elif name == 'unet_encoder':
        from unet_backbone import UnetEncoder
        backbone = UnetEncoder(3)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Conv_block_half(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_block_half, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_block_full(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_block_full, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
      		    nn.BatchNorm2d(ch_out),
         			nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UpsampleBlock(nn.Module):

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, attention=False, full_seg_conv=False):
        super(UpsampleBlock, self).__init__()

        self.attention = attention
        self.full_seg_conv = full_seg_conv

        ch_out = ch_in//2 if ch_out is None else ch_out

        if attention:
            if skip_in != 0:
                self.up = Up_conv(ch_in, ch_out)
                self.att = Attention_block(ch_out, skip_in, ch_out//2)
            else:
                self.up = Up_conv(ch_in, ch_out)
        else:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        if attention:
            if skip_in != 0:
                conv2_in = skip_in + skip_in
            else:
                conv2_in = ch_out
        else:
            conv2_in = ch_out + skip_in

        self.conv_full = Conv_block_full(conv2_in, ch_out)
        self.conv_half = Conv_block_half(conv2_in, ch_out)

    #f - features from skip connection
    def forward(self, x, f=None):

        if self.attention:
            if f is not None:
                x_g = self.up(x)
                x_a = self.att(x_g, f)  # x_a has always f shape
            else:
                # last upsampling - first resnet output has no features - no need for attention gate
                x = self.up(x)
        else:
            x = self.up(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if f is not None:
            if self.attention:
                x = torch.cat([x_a, f], dim=1)
            else:
                x = torch.cat([x, f], dim=1)

        if self.full_seg_conv:
            x = self.conv_full(x)
        else:
            x = self.conv_half(x)

        return x


class ClassificatonHead(nn.Module):
    def __init__(self, channels, features, num_classes, unet_params, exp_head):
        super(ClassificatonHead, self).__init__()
        self.exp_head = exp_head
        # self.conv = nn.Conv2d(channels,num_classes,1)
        self.drop = nn.Dropout(p=0.5)
        self.avgpool = nn.AvgPool2d(features, None, 0)
        self.fc = nn.Linear(channels, num_classes)
        self.gap = {'encoder_layer3' : nn.AvgPool2d(unet_params['encoder']['features'][-2], None, 0),
                     'decoder_skip_layer3' : nn.AvgPool2d(unet_params['decoder']['features'][0], None, 0)}
        

    def forward(self, x):

        # v2. GAP with FC
        if self.exp_head:
            x = self.drop(x)
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.drop(x)
            x = self.fc(x)

        # v1. CONV with GAP
        # x = self.drop(x)
        # x = self.conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.shape[0],-1)
        return x


class Unet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,
                 backbone_name='resnet50',
                 pretrained=True,
                 encoder_freeze=False,
                 segmentation_classes=2,
                 classes_encoder=3,
                 decoder_filters=(256, 128, 64, 32, 16),
                 infer_tensor=torch.zeros(1, 3, 224, 224),
                 shortcut_features='default',
                 attention=False,
                 decoder_use_batchnorm=True,
                 full_seg_conv=False,
                 experimental_head=True):
        super(Unet, self).__init__()

        # SETUP ENCODER BACKBONE
        self.backbone_name = backbone_name
        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(
            backbone_name, pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.replaced_conv1 = True

        #CALCULATE FEATURES SIZES OF SKIP CONNECTIONS
        encoder_params, x, features = self.infer_skip_channels(infer_tensor)
        self.unet_params = {'encoder': encoder_params}
        bb_output_channels = self.unet_params['encoder']['channels_out'][-1]
        shortcut_chs =  self.unet_params['encoder']['channels_out'][:-1]

        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        #BUILD DECODER
        self.upsample_blocks = nn.ModuleList()
        # avoid having more blocks than skip connections
        decoder_filters = decoder_filters[:len(self.shortcut_features)]
        decoder_filters_in = [bb_output_channels] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('upsample_blocks[{}] in: {}   out: {}'.format(
                i, filters_in, filters_out))
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      use_bn=decoder_use_batchnorm,
                                                      attention=attention,
                                                      full_seg_conv=full_seg_conv))
        # last decoder layer without skip connection - to restore full resolution
        self.final_conv = nn.Conv2d(
            decoder_filters[-1], segmentation_classes, kernel_size=(1, 1))
        
        #CALCULATE FEATURES OUTPUT OF DECODER - debug
        decoder_params = self.infer_decoder_channels(x, features)
        self.unet_params['decoder'] = decoder_params
        
        #CLASSIFYING HEAD - ENCODER FEATURES
        self.exp_head = experimental_head
        if self.exp_head:
            in_channels = bb_output_channels + 2 * self.unet_params['encoder']['channels_out'][-2]
            self.classification_head = ClassificatonHead(in_channels, self.unet_params['encoder']['features'][-1],
                                                     classes_encoder, self.unet_params, self.exp_head)
        else:
            self.classification_head = ClassificatonHead(bb_output_channels, self.unet_params['encoder']['features'][-1],
                                                     classes_encoder, self.unet_params, self.exp_head)

        if encoder_freeze:
            self.freeze_encoder()

        self.features = None

    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):

        """ Forward propagation in U-Net. """

        #encoder forward progation 
        x, encoder_features = self.forward_backbone(*input)

        #decoder forward progation 
        decoder_features = {}
        x_seg = x
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = encoder_features[skip_name]
            x_seg = upsample_block(x_seg, skip_features)
            if skip_name in ['layer2','layer3']:
                decoder_features[skip_name] = x_seg

        #decoder full resolution restoration
        x_seg = self.final_conv(x_seg)

        if self.exp_head:

            # merge features for classification
            #features from encoder layer 4
            fencl4 = self.classification_head.avgpool(x).view(x.shape[0], -1)
            fencl4_norm = nn.functional.normalize (fencl4, p = 2, dim = 1)
            #features from encoder layer 3
            fencl3 = self.classification_head.gap['encoder_layer3'](encoder_features['layer3']).view(x.shape[0], -1)
            fencl3_norm = nn.functional.normalize (fencl3, p = 2, dim = 1)
            #features from decoder - after skip connection from layer 3
            fdecl3 = self.classification_head.gap['decoder_skip_layer3'](decoder_features['layer3']).view(x.shape[0], -1)
            fdecl3_norm = nn.functional.normalize (fdecl3, p = 2, dim = 1) 
            
            # max_features = {'enc4': fencl4_norm.max(1).values.tolist(),
            #                 'enc3': fencl3_norm.max(1).values.tolist(),
            #                 'dec3': fdecl3_norm.max(1).values.tolist()}

            #concatenate normalized features
            features = torch.cat([fencl4_norm, fencl3_norm, fdecl3_norm], dim=1)
            x_class = self.classification_head(features)
        else:
            x_class = self.classification_head(x)
        
        return x_seg, x_class

    def forward_backbone(self, x):

        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_decoder_channels(self, x, features):

        decoder_info = {'layers_name': [], 'channels_out': [], 'features': []}
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)
            decoder_info['layers_name'].append(skip_name)
            decoder_info['channels_out'].append(x.shape[1])
            decoder_info['features'].append(x.shape[2])

        x = self.final_conv(x)

        decoder_info['layers_name'].append(skip_name)
        decoder_info['channels_out'].append(x.shape[1])
        decoder_info['features'].append(x.shape[2])
        return decoder_info

    def infer_skip_channels(self, infer_tensor = torch.zeros(1, 3, 224, 224)):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = infer_tensor
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution
        encoder_info = {'layers_name': [], 'channels_out': [], 'features': []}
        features = {None: None} if None in self.shortcut_features else dict()
        
        #first resnet layer has no skip connection
        encoder_info['layers_name'].append('')
        encoder_info['channels_out'].append(0)
        encoder_info['features'].append(0)
        
        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                encoder_info['layers_name'].append(name)
                encoder_info['channels_out'].append(x.shape[1])
                encoder_info['features'].append(x.shape[2])
                features[name] = x
            elif name == self.bb_out_name:
                encoder_info['layers_name'].append(name)
                encoder_info['channels_out'].append(x.shape[1])
                encoder_info['features'].append(x.shape[2])
                break

        return encoder_info, x, features

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param


if __name__ == "__main__":
    
    input_sample = torch.zeros(1, 3, 224, 224)
    # simple test run
    net = Unet(backbone_name='resnet18', pretrained=True, infer_tensor=input_sample, classes=1, classes_encoder=3)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    for _ in range(1):
        with torch.set_grad_enabled(True):
            batch = torch.empty(1, 3, 224, 224).normal_()
            targets = torch.empty(1, 21, 224, 224).normal_()

            out, bb_out = net(batch)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        print(out.shape)