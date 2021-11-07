# Copyright (c) Sangrok Lee and Youngwan Lee (ETRI) All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.utils.registry import Registry
from centermask.layers import MaxPool2d, Linear


SOR_HEAD_REGISTRY = Registry("SOR_HEAD")
SOR_HEAD_REGISTRY.__doc__ = """
Registry for sor heads, which predicts instance ranks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def sor_loss(pred_sor, instances):
    gt_ranks = cat([instance.gt_ranks for instance in instances], dim=0)
    return F.cross_entropy(pred_sor, gt_ranks)


@SOR_HEAD_REGISTRY.register()
class SinglePredictor(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(SinglePredictor, self).__init__()

        # fmt: off
        num_ranks = cfg.MODEL.SOR.NUM_RANKS
        conv_dims = cfg.MODEL.SOR.CONV_DIM
        num_conv = cfg.MODEL.SOR.NUM_CONV
        input_channels = input_shape.channels + 1
        resolution = input_shape.width // 2
        # fmt: on

        self.conv_relus = []
        stride = 1
        for k in range(num_conv):
            if (k+1) == num_conv:
                stride = 2
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=F.relu
            )
            self.add_module("sor_fcn{}".format(k+1), conv)
            self.conv_relus.append(conv)
        self.sor_fc1 = Linear(conv_dims*resolution**2, cfg.MODEL.SOR.DENSE_DIM)
        self.sor_fc2 = Linear(cfg.MODEL.SOR.DENSE_DIM, cfg.MODEL.SOR.OBJ_DIM)
        self.sor = Linear(cfg.MODEL.SOR.OBJ_DIM, num_ranks)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)

        for l in self.conv_relus:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        for l in [self.sor_fc1, self.sor_fc2]:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.sor.weight, mean=0, std=0.01)
        nn.init.constant_(self.sor.bias, 0)
        # print(self)

    def forward(self, x, mask, instances=None, pos=None):
        mask_pool = self.pooling(mask)
        x = torch.cat((x, mask_pool), 1)

        for layer in self.conv_relus:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.sor_fc1(x))
        obj_features = F.relu(self.sor_fc2(x))

        # transformer can be added here

        x = self.sor(obj_features)
        return x


@SOR_HEAD_REGISTRY.register()
class SelfAttentionPredictor(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(SelfAttentionPredictor, self).__init__()

        # fmt: off
        num_ranks = cfg.MODEL.SOR.NUM_RANKS
        conv_dims = cfg.MODEL.SOR.CONV_DIM
        num_conv = cfg.MODEL.SOR.NUM_CONV
        input_channels = input_shape.channels + 1
        resolution = input_shape.width // 2
        # fmt: on

        self.conv_relus = []
        stride = 1
        for k in range(num_conv):
            if (k+1) == num_conv:
                stride = 2
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=F.relu
            )
            self.add_module("sor_fcn{}".format(k+1), conv)
            self.conv_relus.append(conv)
        self.sor_fc1 = Linear(conv_dims*resolution**2, cfg.MODEL.SOR.DENSE_DIM)
        self.sor_fc2 = Linear(cfg.MODEL.SOR.DENSE_DIM, cfg.MODEL.SOR.OBJ_DIM)
        self.sor = Linear(cfg.MODEL.SOR.OBJ_DIM, num_ranks)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
            nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
            dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
            nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
            dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.MODEL.SOR.TRANSFORMER.LAYERS)
        self.obj_dim = cfg.MODEL.SOR.OBJ_DIM

        for l in self.conv_relus:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        for l in [self.sor_fc1, self.sor_fc2]:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.sor.weight, mean=0, std=0.01)
        nn.init.constant_(self.sor.bias, 0)
        for l in self.transformer_encoder.layers:
            for m in l.children():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, mask, instances=None, pos=None):
        mask_pool = self.pooling(mask)
        x = torch.cat((x, mask_pool), 1)

        for layer in self.conv_relus:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.sor_fc1(x))
        obj_features = F.relu(self.sor_fc2(x))

        # transformer can be added here
        nums = [len(p) for p in instances] if instances is not None else [obj_features.shape[0]]

        start = 0
        fusion_features = []
        for i, num in enumerate(nums):
            proposal_fea = obj_features[start:start+nums[i], :]
            start += nums[i]
            fusion = self.transformer_encoder(proposal_fea.reshape(-1, 1, self.obj_dim))
            fusion_features.append(fusion)
        fusion_features = torch.cat(fusion_features, dim=0).squeeze(dim=1)
        x = self.sor(fusion_features)
        return x


@SOR_HEAD_REGISTRY.register()
class SelfAttentionWithPosCat(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(SelfAttentionWithPosCat, self).__init__()

        # fmt: off
        num_ranks = cfg.MODEL.SOR.NUM_RANKS
        conv_dims = cfg.MODEL.SOR.CONV_DIM
        num_conv = cfg.MODEL.SOR.NUM_CONV
        input_channels = input_shape.channels + 1 + 2
        resolution = input_shape.width // 2
        # fmt: on

        self.conv_relus = []
        stride = 1
        for k in range(num_conv):
            if (k+1) == num_conv:
                stride = 2
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=F.relu
            )
            self.add_module("sor_fcn{}".format(k+1), conv)
            self.conv_relus.append(conv)
        self.sor_fc1 = Linear(conv_dims*resolution**2, cfg.MODEL.SOR.DENSE_DIM)
        self.sor_fc2 = Linear(cfg.MODEL.SOR.DENSE_DIM, cfg.MODEL.SOR.OBJ_DIM)
        self.sor = Linear(cfg.MODEL.SOR.OBJ_DIM, num_ranks)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
            nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
            dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
            nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
            dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.MODEL.SOR.TRANSFORMER.LAYERS)
        self.obj_dim = cfg.MODEL.SOR.OBJ_DIM

        for l in self.conv_relus:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        for l in [self.sor_fc1, self.sor_fc2]:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.sor.weight, mean=0, std=0.01)
        nn.init.constant_(self.sor.bias, 0)
        for l in self.transformer_encoder.layers:
            for m in l.children():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, mask, instances=None, pos=None):
        mask_pool = self.pooling(mask)
        x = torch.cat((x, mask_pool, pos), 1)

        for layer in self.conv_relus:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.sor_fc1(x))
        obj_features = F.relu(self.sor_fc2(x))

        # transformer can be added here
        nums = [len(p) for p in instances] if instances is not None else [obj_features.shape[0]]

        start = 0
        fusion_features = []
        for i, num in enumerate(nums):
            proposal_fea = obj_features[start:start+nums[i], :]
            start += nums[i]
            fusion = self.transformer_encoder(proposal_fea.reshape(-1, 1, self.obj_dim))
            fusion_features.append(fusion)
        fusion_features = torch.cat(fusion_features, dim=0).squeeze(dim=1)
        x = self.sor(fusion_features)
        return x

@SOR_HEAD_REGISTRY.register()
class SelfAttentionWithPosConvCat(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(SelfAttentionWithPosConvCat, self).__init__()

        # fmt: off
        num_ranks = cfg.MODEL.SOR.NUM_RANKS
        conv_dims = cfg.MODEL.SOR.CONV_DIM
        num_conv = cfg.MODEL.SOR.NUM_CONV
        input_channels = input_shape.channels + 1 + 2 + 2
        resolution = input_shape.width // 2
        # fmt: on

        self.conv_relus = []
        # self.pos_conv_relus = []
        stride = 1
        for k in range(num_conv):
            if (k+1) == num_conv:
                stride = 2
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=F.relu
            )
            self.add_module("sor_fcn{}".format(k+1), conv)
            self.conv_relus.append(conv)

            # pos_conv = Conv2d(
            #     2 if k == 0 else conv_dims,
            #     conv_dims,
            #     kernel_size=3,
            #     stride=stride,
            #     padding=1,
            #     activation=F.relu
            # )
            # self.add_module("pos_fcn{}".format(k+1), pos_conv)
            # self.pos_conv_relus.append(pos_conv)

        self.sor_fc1 = Linear(conv_dims*resolution**2, cfg.MODEL.SOR.DENSE_DIM)
        self.sor_fc2 = Linear(cfg.MODEL.SOR.DENSE_DIM, cfg.MODEL.SOR.OBJ_DIM)
        self.sor = Linear(cfg.MODEL.SOR.OBJ_DIM, num_ranks)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)
        self.pos_conv = Conv2d(2, 2, kernel_size=3, stride=1, padding=1, activation=F.relu)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
            nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
            dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
            nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
            dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.MODEL.SOR.TRANSFORMER.LAYERS)
        self.obj_dim = cfg.MODEL.SOR.OBJ_DIM

        for l in self.conv_relus:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        nn.init.kaiming_normal_(self.pos_conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.pos_conv.bias, 0)

        for l in [self.sor_fc1, self.sor_fc2]:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.sor.weight, mean=0, std=0.01)
        nn.init.constant_(self.sor.bias, 0)
        for l in self.transformer_encoder.layers:
            for m in l.children():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, mask, instances=None, pos=None):
        mask_pool = self.pooling(mask)
        x = torch.cat((x, mask_pool, pos, self.pos_conv(pos)), 1)

        for layer in self.conv_relus:
            x = layer(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.sor_fc1(x))
        obj_features = F.relu(self.sor_fc2(x))

        # transformer can be added here
        nums = [len(p) for p in instances] if self.training else [obj_features.shape[0]]

        start = 0
        fusion_features = []
        for i, num in enumerate(nums):
            proposal_fea = obj_features[start:start+nums[i], :]
            start += nums[i]
            fusion = self.transformer_encoder(proposal_fea.reshape(-1, 1, self.obj_dim))
            fusion_features.append(fusion)
        fusion_features = torch.cat(fusion_features, dim=0).squeeze(dim=1)
        x = self.sor(fusion_features)
        return x

@SOR_HEAD_REGISTRY.register()
class SinglePredictorWithPos(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(SinglePredictorWithPos, self).__init__()

        # fmt: off
        num_ranks = cfg.MODEL.SOR.NUM_RANKS
        conv_dims = cfg.MODEL.SOR.CONV_DIM
        num_conv = cfg.MODEL.SOR.NUM_CONV
        input_channels = input_shape.channels + 1 + 4
        resolution = input_shape.width // 2
        # fmt: on

        self.conv_relus = []
        stride = 1
        for k in range(num_conv):
            if (k+1) == num_conv:
                stride = 2
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=F.relu
            )
            self.add_module("sor_fcn{}".format(k+1), conv)
            self.conv_relus.append(conv)
        self.sor_fc1 = Linear(conv_dims*resolution**2, cfg.MODEL.SOR.DENSE_DIM)
        self.sor_fc2 = Linear(cfg.MODEL.SOR.DENSE_DIM, cfg.MODEL.SOR.OBJ_DIM)
        self.sor = Linear(cfg.MODEL.SOR.OBJ_DIM, num_ranks)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)
        self.pos_conv = Conv2d(2, 2, kernel_size=3, stride=1, padding=1, activation=F.relu)

        for l in self.conv_relus:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        for l in [self.sor_fc1, self.sor_fc2]:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.sor.weight, mean=0, std=0.01)
        nn.init.constant_(self.sor.bias, 0)
        nn.init.kaiming_normal_(self.pos_conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.pos_conv.bias, 0)
        # print(self)

    def forward(self, x, mask, instances=None, pos=None):
        mask_pool = self.pooling(mask)
        x = torch.cat((x, mask_pool, pos, self.pos_conv(pos)), 1)

        for layer in self.conv_relus:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.sor_fc1(x))
        obj_features = F.relu(self.sor_fc2(x))

        # transformer can be added here

        x = self.sor(obj_features)
        return x

@SOR_HEAD_REGISTRY.register()
class SelfAttentionWithOtherPos(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(SelfAttentionWithOtherPos, self).__init__()

        # fmt: off
        num_ranks = cfg.MODEL.SOR.NUM_RANKS
        conv_dims = cfg.MODEL.SOR.CONV_DIM
        num_conv = cfg.MODEL.SOR.NUM_CONV
        input_channels = input_shape.channels + 1
        resolution = input_shape.width // 2
        # fmt: on

        self.conv_relus = []
        # self.pos_conv_relus = []
        stride = 1
        for k in range(num_conv):
            if (k+1) == num_conv:
                stride = 2
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=F.relu
            )
            self.add_module("sor_fcn{}".format(k+1), conv)
            self.conv_relus.append(conv)

            # pos_conv = Conv2d(
            #     2 if k == 0 else conv_dims,
            #     conv_dims,
            #     kernel_size=3,
            #     stride=stride,
            #     padding=1,
            #     activation=F.relu
            # )
            # self.add_module("pos_fcn{}".format(k+1), pos_conv)
            # self.pos_conv_relus.append(pos_conv)

        self.sor_fc1 = Linear(conv_dims*resolution**2, cfg.MODEL.SOR.DENSE_DIM)
        self.sor_fc2 = Linear(cfg.MODEL.SOR.DENSE_DIM, cfg.MODEL.SOR.OBJ_DIM)
        self.sor = Linear(cfg.MODEL.SOR.OBJ_DIM + 4, num_ranks)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)
        # self.pos_conv = Conv2d(2, 2, kernel_size=3, stride=1, padding=1, activation=F.relu)
        # self.transformer_encoder = nn.TransformerEncoderLayer(
        #     d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
        #     nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
        #     dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        # )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
            nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
            dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.MODEL.SOR.TRANSFORMER.LAYERS)
        self.obj_dim = cfg.MODEL.SOR.OBJ_DIM + 4

        for l in self.conv_relus:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        # nn.init.kaiming_normal_(self.pos_conv.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.constant_(self.pos_conv.bias, 0)

        for l in [self.sor_fc1, self.sor_fc2]:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.sor.weight, mean=0, std=0.01)
        nn.init.constant_(self.sor.bias, 0)
        for l in self.transformer_encoder.layers:
            for m in l.children():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.constant_(m.bias, 0)
        
        self.img_height = cfg.MODEL.SOR.TRANSFORMER.IMG_HEIGHT
        self.img_witdh = cfg.MODEL.SOR.TRANSFORMER.IMG_WIDTH
        


    def forward(self, x, mask, instances=None, pos=None):
        mask_pool = self.pooling(mask)
        x = torch.cat((x, mask_pool), 1)

        for layer in self.conv_relus:
            x = layer(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.sor_fc1(x))
        obj_features = F.relu(self.sor_fc2(x))

        field = 'proposal_boxes' if self.training else 'pred_boxes'
        boxes = torch.cat([instance.get_fields()[field].tensor for instance in instances], 0)
        pos = torch.zeros((boxes.shape[0], 4)).cuda()
        pos[:, 0] = (0.5 * (boxes[:, 0] + boxes[:, 2])) / self.img_witdh
        pos[:, 1] = (0.5 * (boxes[:, 1] + boxes[:, 3])) / self.img_height
        pos[:, 2] = (boxes[:, 2] - boxes[:, 0]) / self.img_witdh
        pos[:, 3] = (boxes[:, 3] + boxes[:, 1]) / self.img_height
        
        obj_features = torch.cat((obj_features, pos), 1)
        
        ## other pos

        # transformer can be added here
        nums = [len(p) for p in instances] if self.training else [obj_features.shape[0]]

        start = 0
        fusion_features = []
        for i, num in enumerate(nums):
            proposal_fea = obj_features[start:start+nums[i], :]
            start += nums[i]
            fusion = self.transformer_encoder(proposal_fea.reshape(-1, 1, self.obj_dim))
            fusion_features.append(fusion)
        fusion_features = torch.cat(fusion_features, dim=0).squeeze(dim=1)
        # print(fusion_features.shape)
        # print(self.sor)
        x = self.sor(fusion_features)
        return x

@SOR_HEAD_REGISTRY.register()
class SelfAttentionWithOtherPosQuant(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(SelfAttentionWithOtherPosQuant, self).__init__()

        # fmt: off
        num_ranks = cfg.MODEL.SOR.NUM_RANKS
        conv_dims = cfg.MODEL.SOR.CONV_DIM
        num_conv = cfg.MODEL.SOR.NUM_CONV
        input_channels = input_shape.channels + 1
        resolution = input_shape.width // 2
        # fmt: on

        self.conv_relus = []
        # self.pos_conv_relus = []
        stride = 1
        for k in range(num_conv):
            if (k+1) == num_conv:
                stride = 2
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=F.relu
            )
            self.add_module("sor_fcn{}".format(k+1), conv)
            self.conv_relus.append(conv)

            # pos_conv = Conv2d(
            #     2 if k == 0 else conv_dims,
            #     conv_dims,
            #     kernel_size=3,
            #     stride=stride,
            #     padding=1,
            #     activation=F.relu
            # )
            # self.add_module("pos_fcn{}".format(k+1), pos_conv)
            # self.pos_conv_relus.append(pos_conv)

        self.sor_fc1 = Linear(conv_dims*resolution**2, cfg.MODEL.SOR.DENSE_DIM)
        self.sor_fc2 = Linear(cfg.MODEL.SOR.DENSE_DIM, cfg.MODEL.SOR.OBJ_DIM)
        self.sor = Linear(cfg.MODEL.SOR.OBJ_DIM, num_ranks)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)
        # self.pos_conv = Conv2d(2, 2, kernel_size=3, stride=1, padding=1, activation=F.relu)
        # self.transformer_encoder = nn.TransformerEncoderLayer(
        #     d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
        #     nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
        #     dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        # )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.SOR.TRANSFORMER.D_MODEL,
            nhead=cfg.MODEL.SOR.TRANSFORMER.N_HEAD,
            dim_feedforward=cfg.MODEL.SOR.TRANSFORMER.D_MODEL * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.MODEL.SOR.TRANSFORMER.LAYERS)
        self.obj_dim = cfg.MODEL.SOR.OBJ_DIM

        for l in self.conv_relus:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        # nn.init.kaiming_normal_(self.pos_conv.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.constant_(self.pos_conv.bias, 0)

        for l in [self.sor_fc1, self.sor_fc2]:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.sor.weight, mean=0, std=0.01)
        nn.init.constant_(self.sor.bias, 0)
        for l in self.transformer_encoder.layers:
            for m in l.children():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.constant_(m.bias, 0)
        
        self.img_height = cfg.MODEL.SOR.TRANSFORMER.IMG_HEIGHT
        self.img_witdh = cfg.MODEL.SOR.TRANSFORMER.IMG_WIDTH
        self.pos_embedding = nn.Parameter(torch.rand(cfg.MODEL.SOR.QUANT_NUM**2, cfg.MODEL.SOR.OBJ_DIM // 2 * 2))
        self.quant_num = cfg.MODEL.SOR.QUANT_NUM

        # self.scale_quant_num = self.quant_num // 2#cfg.MODEL.SOR.SCALE_QUANT_NUM
        # self.scale_embedding = nn.Parameter(torch.rand(self.scale_quant_num**2, cfg.MODEL.SOR.OBJ_DIM // 2))
        

    def forward(self, x, mask, instances=None, pos=None):
        mask_pool = self.pooling(mask)
        x = torch.cat((x, mask_pool), 1)

        for layer in self.conv_relus:
            x = layer(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.sor_fc1(x))
        obj_features = F.relu(self.sor_fc2(x))

        field = 'proposal_boxes' if self.training else 'pred_boxes'
        boxes = torch.cat([instance.get_fields()[field].tensor for instance in instances], 0)
        cx = 0.5 * (boxes[:, 0] + boxes[:, 2]) / (self.img_witdh / self.quant_num)
        cy = 0.5 * (boxes[:, 1] + boxes[:, 3]) / (self.img_height / self.quant_num)
        idx = cx.long() * self.quant_num + cy.long()
        
        pos_features = self.pos_embedding[idx]

        # iw = (boxes[:, 2] - boxes[:, 0]) / (self.img_witdh / self.scale_quant_num)
        # ih = (boxes[:, 3] - boxes[:, 1]) / (self.img_height / self.scale_quant_num)
        # scale_idx = iw.long() * self.scale_quant_num + ih.long()
        # scale_idx[scale_idx >= self.scale_quant_num**2] = self.scale_quant_num**2-1
        
        # if scale_idx.max() >= 16:
        #     with open('tmp.pkl', 'wb') as f:
        #         import pickle
        #         pickle.dump({
        #             "obj_features": obj_features,
        #             "boxes": boxes,
        #         }, f)
        #         print('out of range')
        #         print('-------------------------------------------------------------------------')
        #         exit(0)
        # scale_features = self.scale_embedding[scale_idx]

        
        obj_features = obj_features + pos_features # torch.cat((pos_features, scale_features), 1)
        
        ## other pos
        # with open('tmp.pkl', 'wb') as f:
        #     import pickle
        #     pickle.dump({
        #         "obj_features": obj_features,
        #         "boxes": boxes,
        #     }, f)
        #     print('-------------------------------------------------------------------------')
        #     exit(0)


        # transformer can be added here
        nums = [len(p) for p in instances] if self.training else [obj_features.shape[0]]

        start = 0
        fusion_features = []
        for i, num in enumerate(nums):
            proposal_fea = obj_features[start:start+nums[i], :]
            start += nums[i]
            fusion = self.transformer_encoder(proposal_fea.reshape(-1, 1, self.obj_dim))
            fusion_features.append(fusion)
        fusion_features = torch.cat(fusion_features, dim=0).squeeze(dim=1)
        # print(fusion_features.shape)
        # print(self.sor)
        x = self.sor(fusion_features)
        return x


def build_sor_head(cfg, input_shape):
    name = cfg.MODEL.SOR.NAME
    return SOR_HEAD_REGISTRY.get(name)(cfg, input_shape)
