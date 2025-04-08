import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module, time_dim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.time_dim = time_dim

    def forward(self, input_seq, *args, **kwargs):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        size = list(input_seq.size())
        batch_size = size[0]
        time_steps = size.pop(self.time_dim)
        size_reshape = [batch_size * time_steps] + list(size[1:])
        reshaped_input = input_seq.contiguous().view(size_reshape)

        # Pass the additional arguments to the module
        output = self.module(reshaped_input, *args, **kwargs)

        # reshape output data to original shape
        output_size = [batch_size] + list(output.size())[1:]
        output_size.insert(self.time_dim, time_steps)
        output = output.contiguous().view(output_size)

        return output

class MultiHeadAttention3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, kernel_size=3, stride=1, padding=1):
        super(MultiHeadAttention3D, self).__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        
        self.query_conv = nn.Conv3d(in_channels, out_channels * num_heads, kernel_size, stride, padding)
        self.key_conv = nn.Conv3d(in_channels, out_channels * num_heads, kernel_size, stride, padding)
        self.value_conv = nn.Conv3d(in_channels, out_channels * num_heads, kernel_size, stride, padding)
        
        self.final_conv = nn.Conv3d(out_channels * num_heads, out_channels, kernel_size, stride, padding)
        
    def forward(self, queries, keys, values):
        batch_size, time, channels, depth, height, width = queries.size()
        
        # Flatten the time and batch dimensions together for convolution
        queries = queries.view(batch_size * time, channels, depth, height, width)
        keys = keys.view(batch_size * time, channels, depth, height, width)
        values = values.view(batch_size * time, channels, depth, height, width)
        
        # Project the inputs to queries, keys, and values
        queries = self.query_conv(queries).view(batch_size, time, self.num_heads, self.out_channels, -1)
        keys = self.key_conv(keys).view(batch_size, time, self.num_heads, self.out_channels, -1)
        values = self.value_conv(values).view(batch_size, time, self.num_heads, self.out_channels, -1)
        
        queries = queries.permute(0, 2, 1, 4, 3)  # [batch_size, num_heads, time, depth*height*width, out_channels]
        keys = keys.permute(0, 2, 1, 4, 3)  # [batch_size, num_heads, time, depth*height*width, out_channels]
        values = values.permute(0, 2, 1, 4, 3)  # [batch_size, num_heads, time, depth*height*width, out_channels]
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.out_channels ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum of values
        attn_output = torch.matmul(attention_weights, values)
        attn_output = attn_output.permute(0, 2, 1, 4, 3).contiguous().view(batch_size * time, self.num_heads * self.out_channels, depth, height, width)
        
        # Final projection
        attn_output = self.final_conv(attn_output)
        
        # Reshape back to include time dimension
        attn_output = attn_output.view(batch_size, time, self.out_channels, depth, height, width)
        
        return attn_output, attention_weights


class ScoreLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, query):
        score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))
        return score
    

class ResnetBlock(nn.Module):
    def __init__(self, features, conv3d=False):
        super().__init__()
        self.features = features
        layers = []
        for i in range(2):
            layers += [
                nn.ReflectionPad2d(1) if not conv3d else nn.ReflectionPad3d(1),
                nn.Conv2d(features, features, kernel_size=3) if not conv3d else nn.Conv3d(features, features, kernel_size=3),
                nn.InstanceNorm2d(features) if not conv3d else nn.InstanceNorm3d(features),
            ]
            if i==0:
                layers += [
                    nn.ReLU(True)
                ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return input + self.model(input)
    
class Downsample(nn.Module):
    def __init__(self, features, conv3d=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1) if not conv3d else nn.ReflectionPad3d(1),
            nn.Conv2d(features, features, kernel_size=3, stride=2) if not conv3d else nn.Conv3d(features, features, kernel_size=3, stride=2),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)
    
class Upsample(nn.Module):
    def __init__(self, features, conv3d=False):
        super().__init__()
        layers = [
            nn.ReplicationPad2d(1) if not conv3d else nn.ReplicationPad3d(1),
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=3) if not conv3d else nn.ConvTranspose3d(features, features, kernel_size=4, stride=2, padding=3),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

class Head(nn.Module):
    def __init__(self, in_channels=1, features=64, residuals=9):
        super().__init__()

        mlp = nn.Sequential(*[
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ])
        mlp_id = 0
        setattr(self, 'mlp_%d' % mlp_id, mlp)
        mlp = nn.Sequential(*[
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ])
        mlp_id = 1
        setattr(self, 'mlp_%d' % mlp_id, mlp)
        for mlp_id in range(2, 5):
            mlp = nn.Sequential(*[
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ])
            setattr(self, 'mlp_%d' % mlp_id, mlp)

    def forward(self, feats):
        # if not encode_only:
        #     return(self.model(input))
        # else:
        #     num_patches = 256
        #     return_ids = []
        return_feats = []
        #     feat = input
        #     mlp_id = 0
        for feat_id, feat in enumerate(feats):
            mlp = getattr(self, 'mlp_%d' % feat_id)
            feat = mlp(feat)
            norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
            feat = feat.div(norm + 1e-7)
            return_feats.append(feat)
        return return_feats


class MLPHeads(nn.Module):
    def __init__(self, features=[64, 128, 256, 256, 256]):
        super().__init__()

        for mlp_id, feature in enumerate(features):
            
            mlp = nn.Sequential(*[
                nn.Linear(feature, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ])

            setattr(self, 'mlp_%d' % mlp_id, mlp)
        
    def forward(self, feats):
        
        return_feats = []
        
        for feat_id, feat in enumerate(feats):
            mlp = getattr(self, 'mlp_%d' % feat_id)
            feat = mlp(feat)
            norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
            feat = feat.div(norm + 1e-7)
            return_feats.append(feat)
        return return_feats

class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        self.beta_embed = nn.Embedding(num_classes, num_features)
        # Initialize scale close to 1 and bias as 0.
        nn.init.constant_(self.gamma_embed.weight, 1)
        nn.init.constant_(self.beta_embed.weight, 0)
    
    def forward(self, x, labels):
        out = self.instance_norm(x)
        gamma = self.gamma_embed(labels).unsqueeze(2).unsqueeze(3)
        beta = self.beta_embed(labels).unsqueeze(2).unsqueeze(3)
        return gamma * out + beta

class FiLM(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.features = in_features
        self.num_classes = num_classes
        self.film_gen = nn.Linear(num_classes, in_features * 2)  # Produces [gamma, beta]
    
    def forward(self, x, labels):
        labels = F.one_hot(labels, num_classes=self.num_classes).float()
        film_params = self.film_gen(labels)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

class ConditionalResnetBlock(nn.Module):
    def __init__(self, features, num_classes):
        super().__init__()
        layers = []
        self.features = features
        for i in range(2):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(features, features, kernel_size=3),
                ConditionalInstanceNorm2d(features, num_classes)  
            ]
            if i==0:
                layers += [
                    nn.ReLU(True)
                ]
        self.model = nn.Sequential(*layers)

    def forward(self, input, labels):
        for layer in self.model:
            if hasattr(layer, 'forward') and 'labels' in layer.forward.__code__.co_varnames[:layer.forward.__code__.co_argcount]:
                input = layer(input, labels)
            else:
                input = layer(input)
        return input