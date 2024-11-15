import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from models.depth_anything.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).contiguous().reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


class FeatureProcessor(nn.Module):
    def __init__(self):
        super(FeatureProcessor, self).__init__()

    def forward(self, x):
        return x
        
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024],
                 use_bn=False, use_clstoken=False, localhub=True, **kwargs):
        super(DPT_DINOv2, self).__init__()
        
        encoder = kwargs["encoder"] if "encoder" in kwargs else encoder
        assert encoder in ['vits', 'vitb', 'vitl']
        print("Student Model Encoder Type:", encoder)
        
        if encoder == "vits":
            out_channels = [48, 96, 192, 384]
            features=64
        elif encoder == "vitb":
            features = 128
            out_channels = [96, 192, 384, 768]
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('./models/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder),
                                             source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
            
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        self.pretrained_lr_factor = kwargs["pretrained_lr_factor"] if "pretrained_lr_factor" in kwargs else 1.0
        self.scratch_lr_factor = kwargs["scratch_lr_factor"] if "scratch_lr_factor" in kwargs else 1.0

        # Freeze pretrained weights if required
        if self.pretrained_lr_factor == 0:
            for param in self.pretrained.parameters():
                param.requires_grad = False

        # Initialize a place to store the features
        self.intermediate_features = None

        self.feature_processor = FeatureProcessor()

        self.feature_processor.register_forward_hook(self._hook)

        self.scale, self.shift = 1, 0

    def forward(self, x):
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)

        self.feature_processor(features[-1])  # hook backbone
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1) * self.scale + self.shift

    def _hook(self, module, input, output):
        # This function will store the intermediate output
        self.intermediate_features = output

    def get_lr_params(self, lr):
        """
        Create parameter groups with different learning rates based on the model structure.
        """
        param_conf = []

        # Pretrained model parameters (e.g., VisionTransformer)
        pretrained_params = self.pretrained.parameters()
        pretrained_lr = lr * self.pretrained_lr_factor
        param_conf.append({'params': pretrained_params, 'lr': pretrained_lr})

        # Scratch model parameters
        scratch_params = self.depth_head.parameters()
        scratch_lr = lr * self.scratch_lr_factor
        param_conf.append({'params': scratch_params, 'lr': scratch_lr})

        remaining_params = (param for name, param in self.named_parameters()
                            if "pretrained" not in name and "depth_head" not in name)
        param_conf.append({'params': remaining_params, 'lr': lr})

        return param_conf

    def initialize_weights(self):
        """
        Initialize weights for the layers that are not part of the pretrained model.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def initialize_head_weights(self):
        for module in self.depth_head.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


class DepthAnything(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, **config):
        super().__init__(**config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
    print(model)
    