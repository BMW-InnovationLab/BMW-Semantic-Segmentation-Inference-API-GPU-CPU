# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
# Code partially borrowed from https://github.com/r1ch88/SlowFastNetworks.

__all__ = ['SlowFast', 'slowfast_4x16_resnet50_kinetics400', 'slowfast_8x8_resnet50_kinetics400',
           'slowfast_4x16_resnet101_kinetics400', 'slowfast_8x8_resnet101_kinetics400',
           'slowfast_16x8_resnet101_kinetics400', 'slowfast_16x8_resnet101_50_50_kinetics400',
           'slowfast_4x16_resnet50_custom']

from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

class Bottleneck(HybridBlock):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 strides=1,
                 downsample=None,
                 head_conv=1,
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 layer_name=''):
        super(Bottleneck, self).__init__()

        bottleneck = nn.HybridSequential(prefix=layer_name)
        with bottleneck.name_scope():
            if head_conv == 1:
                self.conv1 = nn.Conv3D(in_channels=inplanes, channels=planes, kernel_size=1, use_bias=False)
                self.bn1 = norm_layer(in_channels=planes, **({} if norm_kwargs is None else norm_kwargs))
            elif head_conv == 3:
                self.conv1 = nn.Conv3D(in_channels=inplanes, channels=planes, kernel_size=(3, 1, 1), padding=(1, 0, 0), use_bias=False)
                self.bn1 = norm_layer(in_channels=planes, **({} if norm_kwargs is None else norm_kwargs))
            else:
                raise ValueError("Unsupported head_conv!")
            self.conv2 = nn.Conv3D(in_channels=planes, channels=planes, kernel_size=(1, 3, 3), strides=(1, strides, strides), padding=(0, 1, 1), use_bias=False)
            self.bn2 = norm_layer(in_channels=planes, **({} if norm_kwargs is None else norm_kwargs))
            self.conv3 = nn.Conv3D(in_channels=planes, channels=planes * self.expansion, kernel_size=1, strides=1, use_bias=False)
            self.bn3 = norm_layer(in_channels=planes * self.expansion, gamma_initializer='zeros', **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.downsample = downsample

    def hybrid_forward(self, F, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.Activation(out + identity, act_type='relu')
        return out


class SlowFast(HybridBlock):
    """SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.
    """
    def __init__(self,
                 nclass,
                 block=Bottleneck,
                 layers=None,
                 num_block_temp_kernel_fast=None,
                 num_block_temp_kernel_slow=None,
                 pretrained=False,
                 pretrained_base=False,
                 feat_ext=False,
                 num_segments=1,
                 num_crop=1,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 frozen_stages=-1,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 alpha=8,
                 beta_inv=8,
                 fusion_conv_channel_ratio=2,
                 fusion_kernel_size=5,
                 width_per_group=64,
                 num_groups=1,
                 slow_temporal_stride=16,
                 fast_temporal_stride=2,
                 slow_frames=4,
                 fast_frames=32,
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 ctx=None,
                 **kwargs):
        super(SlowFast, self).__init__()
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.alpha = alpha
        self.beta_inv = beta_inv
        self.fusion_conv_channel_ratio = fusion_conv_channel_ratio
        self.fusion_kernel_size = fusion_kernel_size
        self.width_per_group = width_per_group
        self.num_groups = num_groups
        self.dim_inner = self.num_groups * self.width_per_group
        self.out_dim_ratio = self.beta_inv // self.fusion_conv_channel_ratio
        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.feat_ext = feat_ext

        with self.name_scope():
            # build fast pathway
            fast = nn.HybridSequential(prefix='fast_')
            with fast.name_scope():
                self.fast_conv1 = nn.Conv3D(in_channels=3, channels=self.width_per_group // self.beta_inv,
                                            kernel_size=(5, 7, 7), strides=(1, 2, 2), padding=(2, 3, 3), use_bias=False)
                self.fast_bn1 = norm_layer(in_channels=self.width_per_group // self.beta_inv,
                                           **({} if norm_kwargs is None else norm_kwargs))
                self.fast_relu = nn.Activation('relu')
                self.fast_maxpool = nn.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding=(0, 1, 1))
            self.fast_res2 = self._make_layer_fast(inplanes=self.width_per_group // self.beta_inv,
                                                   planes=self.dim_inner // self.beta_inv,
                                                   num_blocks=layers[0],
                                                   head_conv=3,
                                                   norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs,
                                                   layer_name='fast_res2_')
            self.fast_res3 = self._make_layer_fast(inplanes=self.width_per_group * 4 // self.beta_inv,
                                                   planes=self.dim_inner * 2 // self.beta_inv,
                                                   num_blocks=layers[1],
                                                   strides=2,
                                                   head_conv=3,
                                                   norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs,
                                                   layer_name='fast_res3_')
            self.fast_res4 = self._make_layer_fast(inplanes=self.width_per_group * 8 // self.beta_inv,
                                                   planes=self.dim_inner * 4 // self.beta_inv,
                                                   num_blocks=layers[2],
                                                   num_block_temp_kernel_fast=num_block_temp_kernel_fast,
                                                   strides=2,
                                                   head_conv=3,
                                                   norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs,
                                                   layer_name='fast_res4_')
            self.fast_res5 = self._make_layer_fast(inplanes=self.width_per_group * 16 // self.beta_inv,
                                                   planes=self.dim_inner * 8 // self.beta_inv,
                                                   num_blocks=layers[3],
                                                   strides=2,
                                                   head_conv=3,
                                                   norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs,
                                                   layer_name='fast_res5_')

            # build lateral connections
            self.lateral_p1 = nn.HybridSequential(prefix='lateral_p1_')
            with self.lateral_p1.name_scope():
                self.lateral_p1.add(nn.Conv3D(in_channels=self.width_per_group // self.beta_inv,
                                              channels=self.width_per_group // self.beta_inv * self.fusion_conv_channel_ratio,
                                              kernel_size=(self.fusion_kernel_size, 1, 1),
                                              strides=(self.alpha, 1, 1),
                                              padding=(self.fusion_kernel_size // 2, 0, 0),
                                              use_bias=False))
                self.lateral_p1.add(norm_layer(in_channels=self.width_per_group // self.beta_inv * self.fusion_conv_channel_ratio,
                                               **({} if norm_kwargs is None else norm_kwargs)))
                self.lateral_p1.add(nn.Activation('relu'))

            self.lateral_res2 = nn.HybridSequential(prefix='lateral_res2_')
            with self.lateral_res2.name_scope():
                self.lateral_res2.add(nn.Conv3D(in_channels=self.width_per_group * 4 // self.beta_inv,
                                                channels=self.width_per_group * 4 // self.beta_inv * self.fusion_conv_channel_ratio,
                                                kernel_size=(self.fusion_kernel_size, 1, 1),
                                                strides=(self.alpha, 1, 1),
                                                padding=(self.fusion_kernel_size // 2, 0, 0),
                                                use_bias=False))
                self.lateral_res2.add(norm_layer(in_channels=self.width_per_group * 4 // self.beta_inv * self.fusion_conv_channel_ratio,
                                                 **({} if norm_kwargs is None else norm_kwargs)))
                self.lateral_res2.add(nn.Activation('relu'))

            self.lateral_res3 = nn.HybridSequential(prefix='lateral_res3_')
            with self.lateral_res3.name_scope():
                self.lateral_res3.add(nn.Conv3D(in_channels=self.width_per_group * 8 // self.beta_inv,
                                                channels=self.width_per_group * 8 // self.beta_inv * self.fusion_conv_channel_ratio,
                                                kernel_size=(self.fusion_kernel_size, 1, 1),
                                                strides=(self.alpha, 1, 1),
                                                padding=(self.fusion_kernel_size // 2, 0, 0),
                                                use_bias=False))
                self.lateral_res3.add(norm_layer(in_channels=self.width_per_group * 8 // self.beta_inv * self.fusion_conv_channel_ratio,
                                                 **({} if norm_kwargs is None else norm_kwargs)))
                self.lateral_res3.add(nn.Activation('relu'))

            self.lateral_res4 = nn.HybridSequential(prefix='lateral_res4_')
            with self.lateral_res4.name_scope():
                self.lateral_res4.add(nn.Conv3D(in_channels=self.width_per_group * 16 // self.beta_inv,
                                                channels=self.width_per_group * 16 // self.beta_inv * self.fusion_conv_channel_ratio,
                                                kernel_size=(self.fusion_kernel_size, 1, 1),
                                                strides=(self.alpha, 1, 1),
                                                padding=(self.fusion_kernel_size // 2, 0, 0),
                                                use_bias=False))
                self.lateral_res4.add(norm_layer(in_channels=self.width_per_group * 16 // self.beta_inv * self.fusion_conv_channel_ratio,
                                                 **({} if norm_kwargs is None else norm_kwargs)))
                self.lateral_res4.add(nn.Activation('relu'))

            # build slow pathway
            slow = nn.HybridSequential(prefix='slow_')
            with slow.name_scope():
                self.slow_conv1 = nn.Conv3D(in_channels=3, channels=self.width_per_group,
                                            kernel_size=(1, 7, 7), strides=(1, 2, 2), padding=(0, 3, 3), use_bias=False)
                self.slow_bn1 = norm_layer(in_channels=self.width_per_group,
                                           **({} if norm_kwargs is None else norm_kwargs))
                self.slow_relu = nn.Activation('relu')
                self.slow_maxpool = nn.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding=(0, 1, 1))
            self.slow_res2 = self._make_layer_slow(inplanes=self.width_per_group + self.width_per_group // self.out_dim_ratio,
                                                   planes=self.dim_inner,
                                                   num_blocks=layers[0],
                                                   head_conv=1,
                                                   norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs,
                                                   layer_name='slow_res2_')
            self.slow_res3 = self._make_layer_slow(inplanes=self.width_per_group * 4 + self.width_per_group * 4 // self.out_dim_ratio,
                                                   planes=self.dim_inner * 2,
                                                   num_blocks=layers[1],
                                                   strides=2,
                                                   head_conv=1,
                                                   norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs,
                                                   layer_name='slow_res3_')
            self.slow_res4 = self._make_layer_slow(inplanes=self.width_per_group * 8 + self.width_per_group * 8 // self.out_dim_ratio,
                                                   planes=self.dim_inner * 4,
                                                   num_blocks=layers[2],
                                                   num_block_temp_kernel_slow=num_block_temp_kernel_slow,
                                                   strides=2,
                                                   head_conv=3,
                                                   norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs,
                                                   layer_name='slow_res4_')
            self.slow_res5 = self._make_layer_slow(inplanes=self.width_per_group * 16 + self.width_per_group * 16 // self.out_dim_ratio,
                                                   planes=self.dim_inner * 8,
                                                   num_blocks=layers[3],
                                                   strides=2,
                                                   head_conv=3,
                                                   norm_layer=norm_layer,
                                                   norm_kwargs=norm_kwargs,
                                                   layer_name='slow_res5_')

            # build classifier
            self.avg = nn.GlobalAvgPool3D()
            self.dp = nn.Dropout(rate=self.dropout_ratio)
            self.feat_dim = self.width_per_group * 32 // self.beta_inv + self.width_per_group * 32
            self.fc = nn.Dense(in_units=self.feat_dim, units=nclass, weight_initializer=init.Normal(sigma=self.init_std), use_bias=True)

            self.initialize(init.MSRAPrelu(), ctx=ctx)

    def hybrid_forward(self, F, x):
        fast_input = F.slice(x, begin=(None, None, 0, None, None), end=(None, None, self.fast_frames, None, None))
        slow_input = F.slice(x, begin=(None, None, self.fast_frames, None, None), end=(None, None, self.fast_frames + self.slow_frames, None, None))

        fast, lateral = self.FastPath(F, fast_input)
        slow = self.SlowPath(F, slow_input, lateral)
        x = F.concat(slow, fast, dim=1)                 # bx2304

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        if self.feat_ext:
            return x

        x = self.dp(x)
        x = self.fc(x)                                  # bxnclass
        return x

    def SlowPath(self, F, x, lateral):
        x = self.slow_conv1(x)                          # bx64x4x112x112, input is bx3x4x224x224
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        pool1 = self.slow_maxpool(x)                    # bx64x4x56x56
        pool1_lat = F.concat(pool1, lateral[0], dim=1)  # bx80x4x56x56

        res2 = self.slow_res2(pool1_lat)                # bx256x4x56x56
        res2_lat = F.concat(res2, lateral[1], dim=1)    # bx320x4x56x56

        res3 = self.slow_res3(res2_lat)                 # bx512x4x28x28
        res3_lat = F.concat(res3, lateral[2], dim=1)    # bx640x4x28x28

        res4 = self.slow_res4(res3_lat)                 # bx1024x4x14x14
        res4_lat = F.concat(res4, lateral[3], dim=1)    # bx1280x4x14x14

        res5 = self.slow_res5(res4_lat)                 # bx2048x4x7x7
        out = self.avg(res5)                            # bx2048x1x1x1
        out = F.squeeze(out, axis=(2, 3, 4))            # bx2048
        return out

    def FastPath(self, F, x):
        lateral = []
        x = self.fast_conv1(x)                          # bx8x32x112x112, input is bx3x32x224x224
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)                    # bx8x32x56x56
        lateral_p = self.lateral_p1(pool1)              # bx16x4x56x56
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)                    # bx32x32x56x56
        lateral_res2 = self.lateral_res2(res2)          # bx64x4x56x56
        lateral.append(lateral_res2)

        res3 = self.fast_res3(res2)                     # bx64x32x28x28
        lateral_res3 = self.lateral_res3(res3)          # bx128x4x28x28
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)                     # bx128x32x14x14
        lateral_res4 = self.lateral_res4(res4)          # bx256x4x14x14
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)                     # bx256x32x7x7
        out = self.avg(res5)                            # bx256x1x1x1
        out = F.squeeze(out, axis=(2, 3, 4))            # bx256
        return out, lateral

    def _make_layer_fast(self,
                         inplanes,
                         planes,
                         num_blocks,
                         num_block_temp_kernel_fast=None,
                         block=Bottleneck,
                         strides=1,
                         head_conv=1,
                         norm_layer=BatchNorm,
                         norm_kwargs=None,
                         layer_name=''):
        downsample = None
        if strides != 1 or inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix=layer_name+'downsample_')
            with downsample.name_scope():
                downsample.add(nn.Conv3D(in_channels=inplanes,
                                         channels=planes * block.expansion,
                                         kernel_size=1,
                                         strides=(1, strides, strides),
                                         use_bias=False))
                downsample.add(norm_layer(in_channels=planes * block.expansion,
                                          **({} if norm_kwargs is None else norm_kwargs)))

        layers = nn.HybridSequential(prefix=layer_name)
        cnt = 0
        with layers.name_scope():
            layers.add(block(inplanes=inplanes,
                             planes=planes,
                             strides=strides,
                             downsample=downsample,
                             head_conv=head_conv,
                             layer_name='block%d_' % cnt))
            inplanes = planes * block.expansion
            cnt += 1
            for _ in range(1, num_blocks):
                if num_block_temp_kernel_fast is not None:
                    if cnt < num_block_temp_kernel_fast:
                        layers.add(block(inplanes=inplanes,
                                         planes=planes,
                                         head_conv=head_conv,
                                         layer_name='block%d_' % cnt))
                    else:
                        layers.add(block(inplanes=inplanes,
                                         planes=planes,
                                         head_conv=1,
                                         layer_name='block%d_' % cnt))
                else:
                    layers.add(block(inplanes=inplanes,
                                     planes=planes,
                                     head_conv=head_conv,
                                     layer_name='block%d_' % cnt))
                cnt += 1
        return layers

    def _make_layer_slow(self,
                         inplanes,
                         planes,
                         num_blocks,
                         num_block_temp_kernel_slow=None,
                         block=Bottleneck,
                         strides=1,
                         head_conv=1,
                         norm_layer=BatchNorm,
                         norm_kwargs=None,
                         layer_name=''):
        downsample = None
        if strides != 1 or inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix=layer_name+'downsample_')
            with downsample.name_scope():
                downsample.add(nn.Conv3D(in_channels=inplanes,
                                         channels=planes * block.expansion,
                                         kernel_size=1,
                                         strides=(1, strides, strides),
                                         use_bias=False))
                downsample.add(norm_layer(in_channels=planes * block.expansion, **({} if norm_kwargs is None else norm_kwargs)))

        layers = nn.HybridSequential(prefix=layer_name)
        cnt = 0
        with layers.name_scope():
            layers.add(block(inplanes=inplanes,
                             planes=planes,
                             strides=strides,
                             downsample=downsample,
                             head_conv=head_conv,
                             layer_name='block%d_' % cnt))
            inplanes = planes * block.expansion
            cnt += 1
            for _ in range(1, num_blocks):
                if num_block_temp_kernel_slow is not None:
                    if cnt < num_block_temp_kernel_slow:
                        layers.add(block(inplanes=inplanes,
                                         planes=planes,
                                         head_conv=head_conv,
                                         layer_name='block%d_' % cnt))
                    else:
                        layers.add(block(inplanes=inplanes,
                                         planes=planes,
                                         head_conv=1,
                                         layer_name='block%d_' % cnt))
                else:
                    layers.add(block(inplanes=inplanes,
                                     planes=planes,
                                     head_conv=head_conv,
                                     layer_name='block%d_' % cnt))
                cnt += 1
        return layers

def slowfast_4x16_resnet50_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                       use_tsn=False, num_segments=1, num_crop=1,
                                       partial_bn=False, feat_ext=False,
                                       root='~/.mxnet/models', ctx=cpu(), **kwargs):
    r"""SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    model = SlowFast(nclass=nclass,
                     layers=[3, 4, 6, 3],
                     pretrained=pretrained,
                     pretrained_base=pretrained_base,
                     feat_ext=feat_ext,
                     num_segments=num_segments,
                     num_crop=num_crop,
                     partial_bn=partial_bn,
                     alpha=8,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=16,
                     fast_temporal_stride=2,
                     slow_frames=4,
                     fast_frames=32,
                     ctx=ctx,
                     **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('slowfast_4x16_resnet50_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def slowfast_8x8_resnet50_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                      use_tsn=False, num_segments=1, num_crop=1,
                                      partial_bn=False, feat_ext=False,
                                      root='~/.mxnet/models', ctx=cpu(), **kwargs):
    r"""SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    model = SlowFast(nclass=nclass,
                     layers=[3, 4, 6, 3],
                     pretrained=pretrained,
                     pretrained_base=pretrained_base,
                     feat_ext=feat_ext,
                     num_segments=num_segments,
                     num_crop=num_crop,
                     partial_bn=partial_bn,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=7,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=8,
                     fast_frames=32,
                     ctx=ctx,
                     **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('slowfast_8x8_resnet50_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def slowfast_4x16_resnet101_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                        use_tsn=False, num_segments=1, num_crop=1,
                                        partial_bn=False, feat_ext=False,
                                        root='~/.mxnet/models', ctx=cpu(), **kwargs):
    r"""SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    model = SlowFast(nclass=nclass,
                     layers=[3, 4, 23, 3],
                     pretrained=pretrained,
                     pretrained_base=pretrained_base,
                     feat_ext=feat_ext,
                     num_segments=num_segments,
                     num_crop=num_crop,
                     partial_bn=partial_bn,
                     alpha=8,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=16,
                     fast_temporal_stride=2,
                     slow_frames=4,
                     fast_frames=32,
                     ctx=ctx,
                     **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('slowfast_4x16_resnet101_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def slowfast_8x8_resnet101_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                       use_tsn=False, num_segments=1, num_crop=1,
                                       partial_bn=False, feat_ext=False,
                                       root='~/.mxnet/models', ctx=cpu(), **kwargs):
    r"""SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    model = SlowFast(nclass=nclass,
                     layers=[3, 4, 23, 3],
                     pretrained=pretrained,
                     pretrained_base=pretrained_base,
                     feat_ext=feat_ext,
                     num_segments=num_segments,
                     num_crop=num_crop,
                     partial_bn=partial_bn,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=8,
                     fast_frames=32,
                     ctx=ctx,
                     **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('slowfast_8x8_resnet101_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def slowfast_16x8_resnet101_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                        use_tsn=False, num_segments=1, num_crop=1,
                                        partial_bn=False, feat_ext=False,
                                        root='~/.mxnet/models', ctx=cpu(), **kwargs):
    r"""SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    model = SlowFast(nclass=nclass,
                     layers=[3, 4, 23, 3],
                     pretrained=pretrained,
                     pretrained_base=pretrained_base,
                     feat_ext=feat_ext,
                     num_segments=num_segments,
                     num_crop=num_crop,
                     partial_bn=partial_bn,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=16,
                     fast_frames=64,
                     ctx=ctx,
                     **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('slowfast_16x8_resnet101_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def slowfast_16x8_resnet101_50_50_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                              use_tsn=False, num_segments=1, num_crop=1,
                                              partial_bn=False, feat_ext=False,
                                              root='~/.mxnet/models', ctx=cpu(), **kwargs):
    r"""SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    model = SlowFast(nclass=nclass,
                     layers=[3, 4, 23, 3],
                     num_block_temp_kernel_fast=6,
                     num_block_temp_kernel_slow=6,
                     pretrained=pretrained,
                     pretrained_base=pretrained_base,
                     feat_ext=feat_ext,
                     num_segments=num_segments,
                     num_crop=num_crop,
                     partial_bn=partial_bn,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=16,
                     fast_frames=64,
                     ctx=ctx,
                     **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('slowfast_16x8_resnet101_50_50_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def slowfast_4x16_resnet50_custom(nclass=400, pretrained=False, pretrained_base=True,
                                  use_tsn=False, num_segments=1, num_crop=1,
                                  partial_bn=False, feat_ext=False, use_kinetics_pretrain=True,
                                  root='~/.mxnet/models', ctx=cpu(), **kwargs):
    r"""SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    model = SlowFast(nclass=nclass,
                     layers=[3, 4, 6, 3],
                     pretrained=pretrained,
                     pretrained_base=pretrained_base,
                     feat_ext=feat_ext,
                     num_segments=num_segments,
                     num_crop=num_crop,
                     partial_bn=partial_bn,
                     alpha=8,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=16,
                     fast_temporal_stride=2,
                     slow_frames=4,
                     fast_frames=32,
                     ctx=ctx,
                     **kwargs)

    if use_kinetics_pretrain and not pretrained:
        from gluoncv.model_zoo import get_model
        kinetics_model = get_model('slowfast_4x16_resnet50_kinetics400', nclass=400, pretrained=True)
        source_params = kinetics_model.collect_params()
        target_params = model.collect_params()
        assert len(source_params.keys()) == len(target_params.keys())

        pretrained_weights = []
        for layer_name in source_params.keys():
            pretrained_weights.append(source_params[layer_name].data())

        for i, layer_name in enumerate(target_params.keys()):
            if i + 2 == len(source_params.keys()):
                # skip the last dense layer
                break
            target_params[layer_name].set_data(pretrained_weights[i])

    model.collect_params().reset_ctx(ctx)

    return model
