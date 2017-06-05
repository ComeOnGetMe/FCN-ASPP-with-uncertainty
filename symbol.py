import mxnet as mx
from resnet import get_resnet


def get_backbone(label_num=20, cell_cap=64, exp="cityscapes", aspp=4, aspp_stride=6, atrous_type='atrous_8',
                 bn_use_global_stats=True, relu_type='relu', aspp_nobias=False):
    res = get_resnet(
        atrous_type=atrous_type,
        bn_use_global_stats=bn_use_global_stats,
        relu_type=relu_type)
    fc1_c_list = []
    for i in range(aspp):
        pad = ((i + 1) * aspp_stride, (i + 1) * aspp_stride)
        dilate = pad
        fc1_c_list.append(mx.symbol.Convolution(data=res, num_filter=label_num*cell_cap, kernel=(3, 3), pad=pad,
                                                dilate=dilate, no_bias=aspp_nobias, name=('fc1_%s_c%d' % (exp, i)),
                                                workspace=4096))
    summ = mx.symbol.ElementWiseSum(*fc1_c_list, name=('fc1_%s' % exp))
    return summ


def get_symbol(label_num=19, ignore_label=255, aspp=4, aspp_stride=6, atrous_type='atrous_8',
               bn_use_global_stats=True, relu_type='relu'):
    fc = get_backbone(label_num, aspp=aspp, aspp_stride=aspp_stride, atrous_type=atrous_type, cell_cap=1,
                      bn_use_global_stats=bn_use_global_stats, relu_type=relu_type)
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='seg_loss', multi_output=True,
                                      use_ignore=True, ignore_label=ignore_label, normalization='valid')
    return softmax


def get_symbol_unc(label_num=20, ignore_label=255, aspp=4, aspp_stride=6, atrous_type='atrous_8',
                   bn_use_global_stats=True, relu_type='relu'):
    fc = get_backbone(label_num, aspp=aspp, aspp_stride=aspp_stride, atrous_type=atrous_type, cell_cap=1,
                      bn_use_global_stats=bn_use_global_stats,
                      relu_type=relu_type)
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='seg_loss', multi_output=True,
                                      use_ignore=True, ignore_label=ignore_label, normalization='valid')
    return softmax


if __name__ == '__main__':
    symbol = get_symbol()
    t = mx.viz.plot_network(symbol, shape={'data': (10, 3, 512, 512)})
    t.render()
