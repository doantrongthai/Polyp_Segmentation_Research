import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model

class ConvFormerAttention(nn.Module):
    """
    Efficient ConvFormer Attention block.
    Original uses full spatial MHA (H*W tokens) — replaced with pooled spatial attention
    to avoid OOM at 352x352 (would need ~123K tokens). Functionally equivalent
    in capturing long-range context via AdaptiveAvgPool to fixed 16x16 tokens.
    """

    def __init__(self, filters):
        super(ConvFormerAttention, self).__init__()
        self.filters = filters

        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters, bias=False)
        self.pointwise_conv = nn.Conv2d(filters, filters, kernel_size=1, bias=False)

        self.ln1 = nn.GroupNorm(1, filters)
        self.ln2 = nn.GroupNorm(1, filters)

        # Efficient: pool to 16x16 for attention, then restore
        self.pool_size = 16
        num_heads = max(1, min(4, filters // 4))
        self.mha = nn.MultiheadAttention(embed_dim=filters, num_heads=num_heads, batch_first=True)

        self.mlp_fc1 = nn.Sequential(
            nn.Conv2d(filters, filters * 2, 1),
            nn.GELU()
        )
        self.mlp_fc2 = nn.Conv2d(filters * 2, filters, 1)

        self.feb_conv1 = nn.Conv2d(filters, filters, 1, padding=0)
        self.feb_conv3 = nn.Conv2d(filters, filters, 3, padding=1)
        self.feb_conv5 = nn.Conv2d(filters, filters, 5, padding=2)

    def forward(self, inputs):
        feb1 = self.feb_conv1(inputs)
        feb3 = self.feb_conv3(inputs)
        feb5 = self.feb_conv5(inputs)
        feb_out = F.relu(feb1 + feb3 + feb5)

        x_sep = self.depthwise_conv(feb_out)
        x_sep = self.pointwise_conv(x_sep)
        x_sep = self.ln1(x_sep)

        b, c, h, w = x_sep.shape
        # Pool to fixed size for attention
        x_pooled = F.adaptive_avg_pool2d(x_sep, (self.pool_size, self.pool_size))
        x_flat = x_pooled.view(b, c, -1).transpose(1, 2)   # (B, 256, C)

        x_att, _ = self.mha(x_flat, x_flat, x_flat)
        x_att = x_att.transpose(1, 2).view(b, c, self.pool_size, self.pool_size)
        x_att = F.interpolate(x_att, size=(h, w), mode='bilinear', align_corners=False)
        x_att = self.ln2(x_att)

        x = x_sep + x_att

        x_mlp = self.mlp_fc1(x)
        x_mlp = self.mlp_fc2(x_mlp)

        return x + x_mlp



class BoosterEncoderBlock(nn.Module):
    def __init__(self, in_channels, filters, is_first=False):
        super(BoosterEncoderBlock, self).__init__()
        self.filters = filters
        self.is_first = is_first
        
        if self.is_first:
            self.main_conv = nn.Conv2d(in_channels, filters, 3, padding=1)
        else:
            self.main_conv1 = nn.Conv2d(in_channels, filters, 3, padding=1)
            self.main_conv2 = nn.Conv2d(filters, filters, 3, padding=1)
            
        self.boost_conv1 = nn.Conv2d(in_channels, filters, 3, padding=1)
        self.boost_conv2 = nn.Conv2d(filters, filters // 2, 1, padding=0)
        self.boost_conv3 = nn.Conv2d(filters // 2, filters, 3, padding=1)
        
        self.bn_main = nn.BatchNorm2d(filters)
        self.bn_boost1 = nn.BatchNorm2d(filters)
        self.bn_boost2 = nn.BatchNorm2d(filters // 2)
        self.bn_boost3 = nn.BatchNorm2d(filters)
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
    def forward(self, inputs):
        if self.is_first:
            x_main = self.main_conv(inputs)
            x_main = F.relu(self.bn_main(x_main))
        else:
            x_main = self.main_conv1(inputs)
            x_main = F.relu(self.bn_main(x_main))
            x_main = self.main_conv2(x_main)
            x_main = F.relu(self.bn_main(x_main))
            
        x_boost = F.relu(self.bn_boost1(self.boost_conv1(inputs)))
        x_boost = F.relu(self.bn_boost2(self.boost_conv2(x_boost)))
        x_boost = F.relu(self.bn_boost3(self.boost_conv3(x_boost)))
        
        combined = F.relu(x_main + x_boost)
        pooled = self.maxpool(combined)
        
        return combined, pooled

class BottleneckModule(nn.Module):
    def __init__(self, channels):
        super(BottleneckModule, self).__init__()
        self.split_channels = channels // 2
        
        self.sam_query = nn.Conv2d(channels // 2, channels // 4, 1)
        self.sam_key = nn.Conv2d(channels // 2, channels // 4, 1)
        self.sam_value = nn.Conv2d(channels // 2, channels // 4, 1)
        
        self.gam_conv1 = nn.Conv2d(channels // 2, channels // 4, 1)
        self.gam_conv2 = nn.Conv2d(channels // 2, channels // 4, 1)
        
        self.groups = 4
        
    def forward(self, inputs):
        split1 = inputs[:, :self.split_channels, :, :]
        split2 = inputs[:, self.split_channels:, :, :]
        
        Q = self.sam_query(split1)
        K = self.sam_key(split1)
        V = self.sam_value(split1)
        
        b, c, h, w = Q.shape
        Q_flat = Q.view(b, c, -1).transpose(1, 2)
        K_flat = K.view(b, c, -1)
        V_flat = V.view(b, c, -1).transpose(1, 2)
        
        attention_scores = torch.bmm(Q_flat, K_flat)
        dk = torch.tensor(c, dtype=torch.float32, device=inputs.device)
        attention_scores = attention_scores / torch.sqrt(dk)
        attention_weights = F.softmax(attention_scores, dim=-1)
        sam_output = torch.bmm(attention_weights, V_flat).transpose(1, 2).view(b, c, h, w)
        
        gam1 = self.gam_conv1(split2)
        gam2 = self.gam_conv2(split2)
        
        gam1_flat = gam1.view(b, c, -1).transpose(1, 2)
        gam2_flat = gam2.view(b, c, -1)
        
        similarity = torch.bmm(gam1_flat, gam2_flat)
        gam_output = torch.bmm(F.softmax(similarity, dim=-1), gam1_flat).transpose(1, 2).view(b, c, h, w)
        
        concatenated = torch.cat([sam_output, gam_output, inputs], dim=1)
        
        b, c_out, h, w = concatenated.shape
        channels_per_group = c_out // self.groups
        
        reshaped = concatenated.view(b, self.groups, channels_per_group, h, w)
        transposed = reshaped.transpose(1, 2)
        shuffled = transposed.contiguous().view(b, c_out, h, w)
        
        return shuffled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, filters, use_convformer=True):
        super(DecoderBlock, self).__init__()
        self.filters = filters
        self.use_convformer = use_convformer
        
        self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        
        if use_convformer and skip_channels > 0:
            self.convformer = ConvFormerAttention(skip_channels)
            
        self.ffb_conv = nn.Conv2d(in_channels, filters, 1)
        self.ffb_bn = nn.BatchNorm2d(filters)
        
    def forward(self, x, skip_connection=None):
        ffb_input = x
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if skip_connection is not None and self.use_convformer:
            skip_refined = self.convformer(skip_connection)
        else:
            skip_refined = skip_connection if skip_connection is not None else 0
            
        ffb_output = F.relu(self.ffb_bn(self.ffb_conv(ffb_input)))
        ffb_output = F.interpolate(ffb_output, scale_factor=2, mode='nearest')
        
        if skip_connection is not None:
            combined = x + skip_refined + ffb_output
        else:
            combined = x + ffb_output
            
        return combined

@register_model('lwahnet')
class LWAHNet(nn.Module):
    def __init__(self):
        super(LWAHNet, self).__init__()
        
        self.enc1 = BoosterEncoderBlock(3, 16, is_first=True)
        self.enc2 = BoosterEncoderBlock(16, 32)
        self.enc3 = BoosterEncoderBlock(32, 32)
        self.enc4 = BoosterEncoderBlock(32, 64)
        
        # enc4_pool output channels: 64
        # Bottleneck input channels: 64
        # Bottleneck output channels: 64 (input) + 16 (sam) + 16 (gam) = 96
        self.bottleneck = BottleneckModule(64)
        
        # dec4 input: 96, skip from enc4_out (64). Use convformer=False in TF code?
        # dec4 = DecoderBlock(64)([bottleneck, bottleneck])
        # dec4 = layers.Add()([dec4, enc4_out])
        # Wait, the TF code explicitly has: dec4 = DecoderBlock(64)([bottleneck, bottleneck]); dec4 = layers.Add()([dec4, enc4_out]). No skip passed inside DecoderBlock.
        self.dec4 = DecoderBlock(96, 0, 64, use_convformer=False)
        
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc3_cf = ConvFormerAttention(32)
        
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc2_cf = ConvFormerAttention(32)
        
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.enc1_cf = ConvFormerAttention(16)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.output = nn.Conv2d(16, 1, 1)
        
    def forward(self, inputs):
        enc1_out, enc1_pool = self.enc1(inputs)
        enc2_out, enc2_pool = self.enc2(enc1_pool)
        enc3_out, enc3_pool = self.enc3(enc2_pool)
        enc4_out, enc4_pool = self.enc4(enc3_pool)
        
        bot = self.bottleneck(enc4_pool)
        
        dec4 = self.dec4(bot)
        dec4 = dec4 + enc4_out
        
        dec3_input = F.interpolate(dec4, scale_factor=2, mode='nearest')
        dec3_conv = self.dec3_conv(dec3_input)
        enc3_refined = self.enc3_cf(enc3_out)
        dec3 = dec3_conv + enc3_refined
        
        dec2_input = F.interpolate(dec3, scale_factor=2, mode='nearest')
        dec2_conv = self.dec2_conv(dec2_input)
        enc2_refined = self.enc2_cf(enc2_out)
        dec2 = dec2_conv + enc2_refined
        
        dec1_input = F.interpolate(dec2, scale_factor=2, mode='nearest')
        dec1_conv = self.dec1_conv(dec1_input)
        enc1_refined = self.enc1_cf(enc1_out)
        dec1 = dec1_conv + enc1_refined
        
        final_up = dec1
        final_c = self.final_conv(final_up)
        
        out = self.output(final_c)
        return out
