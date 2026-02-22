import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F


class Swin_Classifier(nn.Module):
    def __init__(self, in_channels=1, n_class=1, normalization='sigmoid'):
        super(Swin_Classifier, self).__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        feature_size = 96
        # feature_size = 48 # this is for B
        spatial_dims = 3
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=True,
            spatial_dims=spatial_dims
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.head = nn.Linear(feature_size, n_class)
    def forward(self, x_in):
        b = x_in.size()[0]
        x_in = F.interpolate(x_in, size=(256,256,96), mode='trilinear') # this is good
        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = self.head(out.view(b, -1))

        return out


class Swin_Encoder(nn.Module):
    def __init__(self, in_channels=1, n_class=1, normalization='sigmoid'):
        super(Swin_Encoder, self).__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        feature_size = 96
        spatial_dims = 3
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=True,
            spatial_dims=spatial_dims
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

    def forward(self, x_in):
        b = x_in.size()[0]
        x_in = F.interpolate(x_in, size=(256,256,96), mode='trilinear')
        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        h3 = hidden_states_out[3]

        return enc0, enc1, enc2, enc3, dec4, h3
    
class Swin_2E1D_Cls(nn.Module):
    def __init__(self, in_channels=1, n_class=1, normalization='sigmoid'):
        super(Swin_2E1D_Cls, self).__init__()

        feature_size = 96
        spatial_dims = 3
        self.encoder1 = Swin_Encoder(in_channels, n_class, normalization)
        self.encoder2 = Swin_Encoder(in_channels, n_class, normalization)
        norm_name = 'instance'

        self.cnn_enc0 = nn.Conv3d(feature_size*2, feature_size, kernel_size=1, stride=1, padding=0)
        self.cnn_enc1 = nn.Conv3d(feature_size*2, feature_size, kernel_size=1, stride=1, padding=0)
        self.cnn_enc2 = nn.Conv3d(2 * feature_size*2, feature_size*2, kernel_size=1, stride=1, padding=0)
        self.cnn_enc3 = nn.Conv3d(4 * feature_size*2, feature_size*4, kernel_size=1, stride=1, padding=0)
        self.cnn_dec4 = nn.Conv3d(16 * feature_size*2, feature_size*16, kernel_size=1, stride=1, padding=0)
        self.cnn_h3 = nn.Conv3d(8 * feature_size*2, feature_size*8, kernel_size=1, stride=1, padding=0)

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.head = nn.Linear(feature_size, n_class)

    def forward(self, x_in, x_ex):
        b = x_in.size()[0]
        x_in = F.interpolate(x_in, size=(256,256,96), mode='trilinear')
        x_ex = F.interpolate(x_ex, size=(256,256,96), mode='trilinear')

        enc0_in, enc1_in, enc2_in, enc3_in, dec4_in, h3_in = self.encoder1(x_in)
        enc0_ex, enc1_ex, enc2_ex, enc3_ex, dec4_ex, h3_ex = self.encoder2(x_ex)
        
        enc0 = self.cnn_enc0(torch.cat([enc0_in, enc0_ex], dim=1).contiguous())
        enc1 = self.cnn_enc1(torch.cat([enc1_in, enc1_ex], dim=1).contiguous())
        enc2 = self.cnn_enc2(torch.cat([enc2_in, enc2_ex], dim=1).contiguous())
        enc3 = self.cnn_enc3(torch.cat([enc3_in, enc3_ex], dim=1).contiguous())
        dec4 = self.cnn_dec4(torch.cat([dec4_in, dec4_ex], dim=1).contiguous())
        h3   = self.cnn_h3(torch.cat([h3_in, h3_ex], dim=1).contiguous())


        dec3 = self.decoder5(dec4, h3)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = self.head(out.view(b, -1).contiguous())

        return out


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))  # final layer
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)