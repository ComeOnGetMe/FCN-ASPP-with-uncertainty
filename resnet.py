import mxnet as mx
BatchNorm = mx.symbol.BatchNorm
no_bias = True
use_global_stats = True
fix_gamma = False
act_type = 'relu'
bn_momentum = 0.9995
eps = 1e-6


def Conv(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                 dilate=dilate, no_bias=no_bias, name=('%s' % name), workspace=4096)
    return conv


def RELU(data, act_type, name):
    if act_type == 'relu':
        relu = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    elif act_type == 'leaky':
        relu = mx.symbol.LeakyReLU(data=data, act_type=act_type, name=name, slope=0.18)
    else:
        relu = mx.symbol.LeakyReLU(data=data, act_type=act_type, name=name)
    return relu


def Conv_AC(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None):
    conv = Conv(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate, name=name)
    act = RELU(data=conv, act_type=act_type, name=('%s_relu' % name))
    return act


def Conv_BN(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate, name=name)
    bn = BatchNorm(data=conv, name=('bn%s' % suffix), eps=eps, use_global_stats=use_global_stats,
                   momentum=bn_momentum, fix_gamma=fix_gamma)
    return bn


def Conv_BN_AC(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, suffix=''):
    conv = Conv_BN(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate,
                   name=name, suffix=suffix)
    act = RELU(data=conv, act_type=act_type, name=('%s_relu' % name))
    return act


def ResidualFactory_o(data, num_1x1_a, num_3x3_b, num_1x1_c, dilate, suffix):
    branch1 = Conv_BN(data=data,   num_filter=num_1x1_c, kernel=(1, 1), name=('res%s_branch1' % suffix),
                      suffix=('%s_branch1' % suffix), pad=(0, 0))
    branch2a = Conv_BN_AC(data=data,   num_filter=num_1x1_a, kernel=(1, 1), name=('res%s_branch2a' % suffix),
                          suffix=('%s_branch2a' % suffix), pad=(0, 0))
    branch2b = Conv_BN_AC(data=branch2a, num_filter=num_3x3_b, kernel=(3, 3), name=('res%s_branch2b' % suffix),
                          suffix=('%s_branch2b' % suffix), pad=dilate, dilate=dilate)
    branch2c = Conv_BN(data=branch2b, num_filter=num_1x1_c, kernel=(1, 1), name=('res%s_branch2c' % suffix),
                       suffix=('%s_branch2c' % suffix), pad=(0, 0))
    summ = mx.symbol.ElementWiseSum(*[branch2c, branch1], name=('res%s' % suffix))
    summ_ac = RELU(data=summ, act_type=act_type, name=('res%s_relu' % suffix))
    return summ_ac


def ResidualFactory_x(data, num_1x1_a, num_3x3_b, num_1x1_c, dilate, suffix):
    branch2a = Conv_BN_AC(data=data, num_filter=num_1x1_a, kernel=(1, 1), name=('res%s_branch2a' % suffix),
                          suffix=('%s_branch2a' % suffix), pad=(0, 0))
    branch2b = Conv_BN_AC(data=branch2a, num_filter=num_3x3_b, kernel=(3, 3), name=('res%s_branch2b' % suffix),
                          suffix=('%s_branch2b' % suffix), pad=dilate, dilate=dilate)
    branch2c = Conv_BN(data=branch2b, num_filter=num_1x1_c, kernel=(1, 1), name=('res%s_branch2c' % suffix),
                       suffix=('%s_branch2c' % suffix), pad=(0, 0))
    summ = mx.symbol.ElementWiseSum(*[data, branch2c], name=('res%s' % suffix))
    summ_ac = RELU(data=summ, act_type=act_type, name=('res%s_relu' % suffix))
    return summ_ac


def ResidualFactory_d(data, num_1x1_a, num_3x3_b, num_1x1_c, suffix):
    branch1 = Conv_BN(data=data,   num_filter=num_1x1_c, kernel=(1, 1), name=('res%s_branch1' % suffix),
                      suffix=('%s_branch1' % suffix), pad=(0, 0), stride=(2, 2))
    branch2a = Conv_BN_AC(data=data,   num_filter=num_1x1_a, kernel=(1, 1), name=('res%s_branch2a' % suffix),
                          suffix=('%s_branch2a' % suffix), pad=(0, 0), stride=(2, 2))
    branch2b = Conv_BN_AC(data=branch2a, num_filter=num_3x3_b, kernel=(3, 3), name=('res%s_branch2b' % suffix),
                          suffix=('%s_branch2b' % suffix), pad=(1, 1))
    branch2c = Conv_BN(data=branch2b, num_filter=num_1x1_c, kernel=(1, 1), name=('res%s_branch2c' % suffix),
                       suffix=('%s_branch2c' % suffix), pad=(0, 0))
    summ = mx.symbol.ElementWiseSum(*[branch2c, branch1], name=('res%s' % suffix))
    summ_ac = RELU(data=summ, act_type=act_type, name=('res%s_relu' % suffix))
    return summ_ac


def get_resnet(atrous_type, bn_use_global_stats=True, relu_type='relu'):
    global use_global_stats
    global act_type
    use_global_stats = bn_use_global_stats
    act_type = relu_type

    data = mx.symbol.Variable(name="data")

    # group 1
    conv1 = Conv_BN_AC(data=data, num_filter=64, kernel=(7, 7), name='conv1', suffix='conv1', pad=(3, 3), stride=(2, 2))
    conv1 = mx.symbol.Pad(data=conv1, mode='constant', pad_width=(0, 0, 0, 0, 0, 1, 0, 1), constant_value=0)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool1")

    if atrous_type == 'atrous_8':
        # group 2
        res2a = ResidualFactory_o(pool1, 64, 64, 256, (1, 1), '2a')
        res2b = ResidualFactory_x(res2a, 64, 64, 256, (1, 1), '2b')
        res2c = ResidualFactory_x(res2b, 64, 64, 256, (1, 1), '2c')

        # group 3
        res3a = ResidualFactory_d(res2c, 128, 128, 512, '3a')
        res3b1 = ResidualFactory_x(res3a, 128, 128, 512, (1, 1), '3b1')
        res3b2 = ResidualFactory_x(res3b1, 128, 128, 512, (1, 1), '3b2')
        res3b3 = ResidualFactory_x(res3b2, 128, 128, 512, (1, 1), '3b3')

        # group 4
        res4a = ResidualFactory_o(res3b3, 256, 256, 1024, (2, 2), '4a')
        res4b1 = ResidualFactory_x(res4a, 256, 256, 1024, (2, 2), '4b1')
        res4b2 = ResidualFactory_x(res4b1, 256, 256, 1024, (2, 2), '4b2')
        res4b3 = ResidualFactory_x(res4b2, 256, 256, 1024, (2, 2), '4b3')
        res4b4 = ResidualFactory_x(res4b3, 256, 256, 1024, (2, 2), '4b4')
        res4b5 = ResidualFactory_x(res4b4, 256, 256, 1024, (2, 2), '4b5')
        res4b6 = ResidualFactory_x(res4b5, 256, 256, 1024, (2, 2), '4b6')
        res4b7 = ResidualFactory_x(res4b6, 256, 256, 1024, (2, 2), '4b7')
        res4b8 = ResidualFactory_x(res4b7, 256, 256, 1024, (2, 2), '4b8')
        res4b9 = ResidualFactory_x(res4b8, 256, 256, 1024, (2, 2), '4b9')
        res4b10 = ResidualFactory_x(res4b9, 256, 256, 1024, (2, 2), '4b10')
        res4b11 = ResidualFactory_x(res4b10, 256, 256, 1024, (2, 2), '4b11')
        res4b12 = ResidualFactory_x(res4b11, 256, 256, 1024, (2, 2), '4b12')
        res4b13 = ResidualFactory_x(res4b12, 256, 256, 1024, (2, 2), '4b13')
        res4b14 = ResidualFactory_x(res4b13, 256, 256, 1024, (2, 2), '4b14')
        res4b15 = ResidualFactory_x(res4b14, 256, 256, 1024, (2, 2), '4b15')
        res4b16 = ResidualFactory_x(res4b15, 256, 256, 1024, (2, 2), '4b16')
        res4b17 = ResidualFactory_x(res4b16, 256, 256, 1024, (2, 2), '4b17')
        res4b18 = ResidualFactory_x(res4b17, 256, 256, 1024, (2, 2), '4b18')
        res4b19 = ResidualFactory_x(res4b18, 256, 256, 1024, (2, 2), '4b19')
        res4b20 = ResidualFactory_x(res4b19, 256, 256, 1024, (2, 2), '4b20')
        res4b21 = ResidualFactory_x(res4b20, 256, 256, 1024, (2, 2), '4b21')
        res4b22 = ResidualFactory_x(res4b21, 256, 256, 1024, (2, 2), '4b22')

        # group 5
        res5a = ResidualFactory_o(res4b22, 512, 512, 2048, (4, 4), '5a')
        res5b = ResidualFactory_x(res5a, 512, 512, 2048, (4, 4), '5b')
        res5c = ResidualFactory_x(res5b, 512, 512, 2048, (4, 4), '5c')
        return res5c
    elif atrous_type == 'atrous_4':
        # group 2
        res2a = ResidualFactory_o(pool1, 64, 64, 256, (1, 1), '2a')
        res2b = ResidualFactory_x(res2a, 64, 64, 256, (1, 1), '2b')
        res2c = ResidualFactory_x(res2b, 64, 64, 256, (1, 1), '2c')

        # group 3
        res3a = ResidualFactory_o(res2c, 128, 128, 512, (2, 2), '3a')
        res3b1 = ResidualFactory_x(res3a, 128, 128, 512, (2, 2), '3b1')
        res3b2 = ResidualFactory_x(res3b1, 128, 128, 512, (2, 2), '3b2')
        res3b3 = ResidualFactory_x(res3b2, 128, 128, 512, (2, 2), '3b3')

        # group 4
        res4a = ResidualFactory_o(res3b3, 256, 256, 1024, (4, 4), '4a')
        res4b1 = ResidualFactory_x(res4a, 256, 256, 1024, (4, 4), '4b1')
        res4b2 = ResidualFactory_x(res4b1, 256, 256, 1024, (4, 4), '4b2')
        res4b3 = ResidualFactory_x(res4b2, 256, 256, 1024, (4, 4), '4b3')
        res4b4 = ResidualFactory_x(res4b3, 256, 256, 1024, (4, 4), '4b4')
        res4b5 = ResidualFactory_x(res4b4, 256, 256, 1024, (4, 4), '4b5')
        res4b6 = ResidualFactory_x(res4b5, 256, 256, 1024, (4, 4), '4b6')
        res4b7 = ResidualFactory_x(res4b6, 256, 256, 1024, (4, 4), '4b7')
        res4b8 = ResidualFactory_x(res4b7, 256, 256, 1024, (4, 4), '4b8')
        res4b9 = ResidualFactory_x(res4b8, 256, 256, 1024, (4, 4), '4b9')
        res4b10 = ResidualFactory_x(res4b9, 256, 256, 1024, (4, 4), '4b10')
        res4b11 = ResidualFactory_x(res4b10, 256, 256, 1024, (4, 4), '4b11')
        res4b12 = ResidualFactory_x(res4b11, 256, 256, 1024, (4, 4), '4b12')
        res4b13 = ResidualFactory_x(res4b12, 256, 256, 1024, (4, 4), '4b13')
        res4b14 = ResidualFactory_x(res4b13, 256, 256, 1024, (4, 4), '4b14')
        res4b15 = ResidualFactory_x(res4b14, 256, 256, 1024, (4, 4), '4b15')
        res4b16 = ResidualFactory_x(res4b15, 256, 256, 1024, (4, 4), '4b16')
        res4b17 = ResidualFactory_x(res4b16, 256, 256, 1024, (4, 4), '4b17')
        res4b18 = ResidualFactory_x(res4b17, 256, 256, 1024, (4, 4), '4b18')
        res4b19 = ResidualFactory_x(res4b18, 256, 256, 1024, (4, 4), '4b19')
        res4b20 = ResidualFactory_x(res4b19, 256, 256, 1024, (4, 4), '4b20')
        res4b21 = ResidualFactory_x(res4b20, 256, 256, 1024, (4, 4), '4b21')
        res4b22 = ResidualFactory_x(res4b21, 256, 256, 1024, (4, 4), '4b22')

        # group 5
        res5a = ResidualFactory_o(res4b22, 512, 512, 2048, (8, 8), '5a')
        res5b = ResidualFactory_x(res5a, 512, 512, 2048, (8, 8), '5b')
        res5c = ResidualFactory_x(res5b, 512, 512, 2048, (8, 8), '5c')
        return res5c
    elif atrous_type == 'atrous_16':
        # group 2
        res2a = ResidualFactory_o(pool1, 64, 64, 256, (1, 1), '2a')
        res2b = ResidualFactory_x(res2a, 64, 64, 256, (1, 1), '2b')
        res2c = ResidualFactory_x(res2b, 64, 64, 256, (1, 1), '2c')

        # group 3
        res3a = ResidualFactory_d(res2c, 128, 128, 512, '3a')
        res3b1 = ResidualFactory_x(res3a, 128, 128, 512, (1, 1), '3b1')
        res3b2 = ResidualFactory_x(res3b1, 128, 128, 512, (1, 1), '3b2')
        res3b3 = ResidualFactory_x(res3b2, 128, 128, 512, (1, 1), '3b3')

        # group 4
        res4a = ResidualFactory_d(res3b3, 256, 256, 1024, '4a')
        res4b1 = ResidualFactory_x(res4a, 256, 256, 1024, (1, 1), '4b1')
        res4b2 = ResidualFactory_x(res4b1, 256, 256, 1024, (1, 1), '4b2')
        res4b3 = ResidualFactory_x(res4b2, 256, 256, 1024, (1, 1), '4b3')
        res4b4 = ResidualFactory_x(res4b3, 256, 256, 1024, (1, 1), '4b4')
        res4b5 = ResidualFactory_x(res4b4, 256, 256, 1024, (1, 1), '4b5')
        res4b6 = ResidualFactory_x(res4b5, 256, 256, 1024, (1, 1), '4b6')
        res4b7 = ResidualFactory_x(res4b6, 256, 256, 1024, (1, 1), '4b7')
        res4b8 = ResidualFactory_x(res4b7, 256, 256, 1024, (1, 1), '4b8')
        res4b9 = ResidualFactory_x(res4b8, 256, 256, 1024, (1, 1), '4b9')
        res4b10 = ResidualFactory_x(res4b9, 256, 256, 1024, (1, 1), '4b10')
        res4b11 = ResidualFactory_x(res4b10, 256, 256, 1024, (1, 1), '4b11')
        res4b12 = ResidualFactory_x(res4b11, 256, 256, 1024, (1, 1), '4b12')
        res4b13 = ResidualFactory_x(res4b12, 256, 256, 1024, (1, 1), '4b13')
        res4b14 = ResidualFactory_x(res4b13, 256, 256, 1024, (1, 1), '4b14')
        res4b15 = ResidualFactory_x(res4b14, 256, 256, 1024, (1, 1), '4b15')
        res4b16 = ResidualFactory_x(res4b15, 256, 256, 1024, (1, 1), '4b16')
        res4b17 = ResidualFactory_x(res4b16, 256, 256, 1024, (1, 1), '4b17')
        res4b18 = ResidualFactory_x(res4b17, 256, 256, 1024, (1, 1), '4b18')
        res4b19 = ResidualFactory_x(res4b18, 256, 256, 1024, (1, 1), '4b19')
        res4b20 = ResidualFactory_x(res4b19, 256, 256, 1024, (1, 1), '4b20')
        res4b21 = ResidualFactory_x(res4b20, 256, 256, 1024, (1, 1), '4b21')
        res4b22 = ResidualFactory_x(res4b21, 256, 256, 1024, (1, 1), '4b22')

        # group 5
        res5a = ResidualFactory_o(res4b22, 512, 512, 2048, (2, 2), '5a')
        res5b = ResidualFactory_x(res5a, 512, 512, 2048, (2, 2), '5b')
        res5c = ResidualFactory_x(res5b, 512, 512, 2048, (2, 2), '5c')
        return res5c

    else:
        pass
