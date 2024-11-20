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
    
