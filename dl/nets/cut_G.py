import torch
import torch.nn as nn

from nets.downsample_layer import Downsample
from nets.upsample_layer import Upsample
from nets.resnet_block import ResnetBlock

class Generator(nn.Module):
    def __init__(self, in_channels=1, features=64, residuals=9, conv3d=False):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3) if not conv3d else nn.ReflectionPad3d(3),
            nn.Conv2d(in_channels, features, kernel_size=7) if not conv3d else nn.Conv3d(in_channels, features, kernel_size=7),
            nn.InstanceNorm2d(features) if not conv3d else nn.InstanceNorm3d(features),
            nn.ReLU(True)
        ]
        features_prev = features
        for i in range(2):
            features *= 2
            layers += [
                nn.Conv2d(features_prev, features, kernel_size=3, stride=1, padding=1) if not conv3d else nn.Conv3d(features_prev, features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(features) if not conv3d else nn.InstanceNorm3d(features),
                nn.ReLU(True),
                Downsample(features, conv3d=conv3d)
                # nn.ReflectionPad2d(1),
                # nn.Conv2d(features, features, kernel_size=3, stride=2)
            ]
            features_prev = features
        for i in range(residuals):
            layers += [ResnetBlock(features_prev, conv3d=conv3d)]
        for i in range(2):
            features //= 2
            layers += [
                # nn.ReplicationPad2d(1),
                # nn.ConvTranspose2d(features_prev, features_prev, kernel_size=4, stride=2, padding=3),
                Upsample(features_prev, conv3d=conv3d),
                nn.Conv2d(features_prev, features, kernel_size=3, stride=1, padding=1) if not conv3d else nn.Conv3d(features_prev, features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(features) if not conv3d else nn.InstanceNorm3d(features),
                nn.ReLU(True)
            ]
            features_prev = features
        layers += [
            nn.ReflectionPad2d(3) if not conv3d else nn.ReflectionPad3d(3),
            nn.Conv2d(features_prev, in_channels, kernel_size=7) if not conv3d else nn.Conv3d(features_prev, in_channels, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input, encode_only=False, patch_ids=None):
        if not encode_only:
            return(self.model(input))
        else:
            num_patches = 256
            return_ids = []
            return_feats = []
            feat = input
            mlp_id = 0
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in [0, 4, 8, 12, 16]:
                    # print("F1", feat.shape)
                    B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                    feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
                    # print("F2", feat_reshape.shape)
                    if patch_ids is not None:
                        patch_id = patch_ids[mlp_id]
                        mlp_id += 1
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[1]) #, device=config.DEVICE
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))] # .to(patch_ids.device)
                        return_ids.append(patch_id)
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
                    # print("F3", x_sample.shape)
                    # print()
                    # mlp = getattr(self, 'mlp_%d' % mlp_id)
                    # x_sample = mlp(x_sample)
                    # mlp_id += 1
                    # norm = x_sample.pow(2).sum(1, keepdim=True).pow(1. / 2)
                    # x_sample = x_sample.div(norm + 1e-7)

                    return_feats.append(x_sample)
            return return_feats, return_ids

def test():
    x = torch.randn((5, 3, 256, 256)).to()
    print(x.shape)
    G = Generator().to()
    feat_k_pool, sample_ids = G(x, encode_only=True, patch_ids=None)
    feat_q_pool, _ = G(x, encode_only=True, patch_ids=sample_ids)
    print(len(feat_k_pool))
    # print(feat_q_pool.shape)

if __name__ == "__main__":
    test()