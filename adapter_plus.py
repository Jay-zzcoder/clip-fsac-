import torch
from torch import nn as nn
import timm
import numpy as np
import math



def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        
        
        
class Linear_Projection(nn.Module):
    def __init__(self, in_channel, out_channel, activation=None,):
        super(Linear_Projection, self).__init__()
        mid_channel=1024
        self.fc1 = nn.Linear(in_channel, mid_channel)
        self.fc2 = nn.Linear(mid_channel, out_channel)
        self.fc = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, activation, dropout_prob=0.0):
        """
        Multi-Layer Perceptron (MLP) module.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden units in the MLP.
            activation (nn.Module): Activation function (e.g., nn.ReLU, nn.LeakyReLU, etc.).
            dropout_prob (float, optional): Dropout probability. Defaults to 0.0.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(init_weight)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the MLP.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TextHead(nn.Module):
    '''
    Text head module.

    Args:
        feature_dim (int): Input feature dimension.
        mlp_layers (int): Number of MLP layers.
        mlp_ratio (int): Multiplier to compute the hidden layer dimension.
        out_dim (int): Output dimension.
        act_fn (nn.Module): Activation function.
        drop (float): Dropout probability.
    '''

    def __init__(self, feature_dim, mlp_layers=1, mlp_ratio=4, out_dim=256, act_fn=nn.GELU, drop=0., res=False, no_head=False):
        super(TextHead, self).__init__()
        self.mlp_layers = mlp_layers

        self.mlps = nn.ModuleList([
            MLP(feature_dim, feature_dim * mlp_ratio, act_fn, drop)
            for _ in range(mlp_layers)
        ])

        self.activation = act_fn()
        self.linear = nn.Linear(feature_dim, out_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=drop)
        self.res = res
        self.no_head = no_head
        self.apply(init_weight)

    def forward(self, feature, atten=None):
        if self.no_head:
            return feature
        origin = feature
        for mlp_layer in self.mlps:
            feature = mlp_layer(feature)
            feature = self.activation(feature)
            feature = self.dropout(feature)

        feature = self.linear(feature)
        if self.res:
            if atten is not None:
                feature = feature + origin + atten
            else:
                #feature = 0.4*feature + 0.6*origin #mvtec
                feature = 0.1*feature + 0.9*origin #visa
            
        return feature
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()



class ImageHead(nn.Module):
    '''
    Image head module.

    Args:
        feature_dim (int): Input feature dimension.
        mlp_layers (int): Number of MLP layers.
        mlp_ratio (int): Multiplier to compute the hidden layer dimension.
        out_dim (int): Output dimension.
        act_fn (nn.Module): Activation function.
        drop (float): Dropout probability.
    '''

    def __init__(self, feature_dim, mlp_layers=2, mlp_ratio=4, out_dim=256, act_fn=nn.GELU, drop=0., res=False, no_head=False, project_out=True):
        super(ImageHead, self).__init__()
        self.mlp_layers = mlp_layers

        self.mlps = nn.ModuleList([
            MLP(feature_dim, feature_dim * mlp_ratio, act_fn, drop)
            for _ in range(mlp_layers)
        ])
        
        self.scale = feature_dim ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(drop)
        
        self.activation = act_fn()
        self.linear = nn.Linear(feature_dim, out_dim)
        self.attention_linear = nn.Sequential(
                  nn.Linear(feature_dim, out_dim),
                  nn.Sigmoid()
                )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=drop)
        self.res = res
        self.no_head = no_head
        
        self.to_out_image = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
            nn.Dropout(drop)
        ) 
        self.to_out_text = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
            nn.Dropout(drop)
        ) 
        self.apply(init_weight)

    def forward(self, feature, text_feature):
        if self.no_head:
            return feature
        origin = feature
        attention_feature = self.image_to_text_cross_attention(feature, text_feature)
        t_ = self.text_to_image_cross_attention(attention_feature, text_feature)
        for mlp_layer in self.mlps:
            feature = mlp_layer(feature)
            feature = self.activation(feature)
            feature = self.dropout(feature)
        
        #atten = self.attention_linear(feature)
        features = self.linear(feature)
        if self.res:
            #features = 0.4*features + 0.6*origin + attention_feature   #mvtec
            features = 0.1*features + 0.9*origin + attention_feature   #visa
            #features = features + origin 
        return features, t_
        
    def image_to_text_cross_attention(self, image_feature, text_feature):
        #print("image_feature shape: ", image_feature.shape)
        #print("text_feature shape: ", text_feature.shape)
        dots = torch.matmul(image_feature, text_feature.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, text_feature)
        out = self.to_out_text(out)
        #print("out shape: ", out.shape)
        return out
        
    def text_to_image_cross_attention(self, image_feature, text_feature):
        #print("image_feature shape: ", image_feature.shape)
        #print("text_feature shape: ", text_feature.shape)
        dots = torch.matmul(text_feature, image_feature.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, image_feature)
        out = self.to_out_image(out)
        #print("out shape: ", out.shape)
        return out
        

class NewLoss(nn.Module):
    def __init__(self):
        super(NewLoss, self).__init__()
        self.th = 1
        self.th_ = -1
        self.t = 0.9

    def forward(self, img_features, noise_image_features, text_features):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        noise_image_features = noise_image_features / noise_image_features.norm(dim=-1, keepdim=True)

        N = img_features.shape[0]

        sim = img_features @ text_features.T
        noise_sim = noise_image_features @ text_features.T

        normal_score = torch.exp(sim[:, 0] / self.t)  # 1
        abnormal_score = torch.exp(sim[:, 1] / self.t)  # 2
        noise_normal_score = torch.exp(noise_sim[:, 0] / self.t)  # 3
        noise_abnormal_score = torch.exp(noise_sim[:, 1] / self.t)  # 4

        a = normal_score + noise_abnormal_score
        b = abnormal_score + noise_normal_score

        loss = -torch.log(0.5 * a / (a + b)).sum() / N

        loss1 = torch.max(self.th - a, torch.tensor(0)).sum() / N
        loss2 = torch.max(-self.th_ + b, torch.tensor(0)).sum() / N

        loss3 = -(torch.log(normal_score / (normal_score + abnormal_score)) + torch.log(
            noise_abnormal_score / (noise_abnormal_score + noise_normal_score))).sum()
        loss4 = -(torch.log(normal_score / (normal_score + noise_normal_score)) + torch.log(
            noise_abnormal_score / (noise_abnormal_score + abnormal_score))).sum()

        return (loss3 + loss4) / (2 * N)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.T = temperature

    def forward(self, img_features, noise_idxs, text_feature):
        img_features = nn.functional.normalize(img_features, dim=1)
        text_feature = nn.functional.normalize(text_feature, dim=1)

        noise_features = img_features[noise_idxs]
        noise_features = nn.functional.normalize(noise_features, dim=1)

        sim_noise = noise_features @ text_feature.t()
        sim_noise /= self.T
        sim_noise = torch.exp(sim_noise)
        sim_noise = sim_noise.sum()

        sim = img_features @ text_feature.t()
        sim /= self.T
        sim = torch.exp(sim)
        sim = sim.sum()

        loss = -torch.log(sim_noise / sim)
        return loss


class Patchfiy(nn.Module):
    def __init__(self, layers_to_extract_from, patchsize, out_dim, stride):
        super(Patchfiy, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.patchsize = patchsize
        self.out_dim = out_dim
        self.stride = stride

    def forward(self, features):
        features = self.extract_layer_features(features)
        features = self.patchify(features)
        return self.features_pooling(features)

    def extract_layer_features(self, features):
        features = [features[layer] for layer in self.layers_to_extract_from]
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
        return features

    def patchify(self, features):
        _features = []
        for feature in features:
            padding = int((self.patchsize - 1) / 2)
            unfolder = torch.nn.Unfold(
                kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
            )
            unfolded_feature = unfolder(feature)
            unfolded_feature = unfolded_feature.reshape(
                *feature.shape[:2], self.patchsize, self.patchsize, -1
            )
            _features.append(unfolded_feature.permute(0, 4, 1, 2, 3))
        return _features


def features_pooling(self, features):
    features = [x.reshape(-1, *x.shape[-3:]) for x in features]
    _features = []
    for feature in features:
        feature = feature.reshape(len(feature), 1, -1)
        feature = torch.nn.functional.adaptive_avg_pool1d(feature, self.out_dim).squeeze(1)
        _features.append(feature)
    features = torch.stack(_features, dim=1)
    features = features.reshape(len(features), 1, -1)
    features = torch.nn.functional.adaptive_avg_pool1d(features, self.out_dim)
    return features.reshape(len(features), -1)


