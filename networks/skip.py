import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import *

from .non_local_dot_product import NONLocalBlock2D



def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if i>1:
            deeper.add(NONLocalBlock2D(in_channels=num_channels_down[i]))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))


        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    # 额外增加标准差输出层
    std_layer = nn.Conv2d(num_output_channels, num_output_channels, kernel_size=1)
    return model,std_layer

# VDIP 生成器类
class VDIPGenerator(nn.Module):
    """ VDIP 生成器，输出均值和标准差 """
    def __init__(self, input_depth, output_channels, num_channels_down, num_channels_up, num_channels_skip,
                 upsample_mode='bilinear', need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU'):
        super(VDIPGenerator, self).__init__()

        # 原始 DIP 生成器（用于生成均值）
        self.mean_generator, self.std_layer = skip(
            input_depth, output_channels,
            num_channels_down=num_channels_down,
            num_channels_up=num_channels_up,
            num_channels_skip=num_channels_skip,
            upsample_mode='nearest',
            need_sigmoid=True,
            need_bias=True,
            pad='reflection',
            act_fun='LeakyReLU'
        )

    def forward(self, z):
        mean_output = self.mean_generator(z)  # 生成均值
        std_output = self.std_layer(mean_output)  # 生成标准差
        return mean_output, F.softplus(std_output)  # 确保标准差为正

if __name__ == '__main__':
    from torchsummary import summary

    net = skip(8, 1, num_channels_down=[128, 128, 128, 128, 128], num_channels_up=[128, 128, 128, 128, 128],
               num_channels_skip=[16, 16, 16, 16, 16],
               upsample_mode='bilinear', need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

    print(net)