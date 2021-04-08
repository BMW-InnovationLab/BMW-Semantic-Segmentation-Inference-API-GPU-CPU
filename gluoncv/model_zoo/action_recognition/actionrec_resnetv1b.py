# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from ..resnetv1b import resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b, resnet152_v1b

__all__ = ['resnet18_v1b_sthsthv2', 'resnet34_v1b_sthsthv2', 'resnet50_v1b_sthsthv2',
           'resnet101_v1b_sthsthv2', 'resnet152_v1b_sthsthv2', 'resnet18_v1b_kinetics400',
           'resnet34_v1b_kinetics400', 'resnet50_v1b_kinetics400', 'resnet101_v1b_kinetics400',
           'resnet152_v1b_kinetics400', 'resnet50_v1b_ucf101', 'resnet50_v1b_hmdb51',
           'resnet50_v1b_custom']

class ActionRecResNetV1b(HybridBlock):
    r"""ResNet models for video action recognition

    Parameters
    ----------
    depth : int, number of layers in a ResNet model
    nclass : int, number of classes
    pretrained_base : bool, load pre-trained weights or not
    dropout_ratio : float, add a dropout layer to prevent overfitting on small datasets, such as UCF101
    init_std : float, standard deviation value when initialize the last classification layer
    feat_dim : int, feature dimension. Default is 4096 for VGG16 network
    num_segments : int, number of segments used
    num_crop : int, number of crops used during evaluation. Default choice is 1, 3 or 10

    Input: a single video frame or N images from N segments when num_segments > 1
    Output: a single predicted action label
    """
    def __init__(self, depth, nclass, pretrained_base=True,
                 dropout_ratio=0.5, init_std=0.01,
                 feat_dim=2048, num_segments=1, num_crop=1,
                 partial_bn=False, **kwargs):
        super(ActionRecResNetV1b, self).__init__()

        if depth == 18:
            pretrained_model = resnet18_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 34:
            pretrained_model = resnet34_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 50:
            pretrained_model = resnet50_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 101:
            pretrained_model = resnet101_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 152:
            pretrained_model = resnet152_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        else:
            print('No such ResNet configuration for depth=%d' % (depth))

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.feat_dim = 512 * self.expansion
        self.num_segments = num_segments
        self.num_crop = num_crop

        with self.name_scope():
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.relu = pretrained_model.relu
            self.maxpool = pretrained_model.maxpool
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
            self.avgpool = pretrained_model.avgpool
            self.flat = pretrained_model.flat
            self.drop = nn.Dropout(rate=self.dropout_ratio)
            self.output = nn.Dense(units=nclass, in_units=self.feat_dim,
                                   weight_initializer=init.Normal(sigma=self.init_std))
            self.output.initialize()

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.drop(x)

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.output(x)
        return x

def resnet18_v1b_sthsthv2(nclass=174, pretrained=False, pretrained_base=True,
                          use_tsn=False, partial_bn=False,
                          num_segments=1, num_crop=1, root='~/.mxnet/models',
                          ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=18,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet18_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet34_v1b_sthsthv2(nclass=174, pretrained=False, pretrained_base=True,
                          use_tsn=False, partial_bn=False,
                          num_segments=1, num_crop=1, root='~/.mxnet/models',
                          ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=34,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet34_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_sthsthv2(nclass=174, pretrained=False, pretrained_base=True,
                          use_tsn=False, partial_bn=False,
                          num_segments=1, num_crop=1, root='~/.mxnet/models',
                          ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=50,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet50_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet101_v1b_sthsthv2(nclass=174, pretrained=False, pretrained_base=True,
                           use_tsn=False, partial_bn=False,
                           num_segments=1, num_crop=1, root='~/.mxnet/models',
                           ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=101,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet101_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet152_v1b_sthsthv2(nclass=174, pretrained=False, pretrained_base=True,
                           use_tsn=False, partial_bn=False,
                           num_segments=1, num_crop=1, root='~/.mxnet/models',
                           ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=152,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet152_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet18_v1b_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=18,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet18_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet34_v1b_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=34,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet34_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=50,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet50_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet101_v1b_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                              use_tsn=False, partial_bn=False,
                              num_segments=1, num_crop=1, root='~/.mxnet/models',
                              ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=101,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet101_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet152_v1b_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                              use_tsn=False, partial_bn=False,
                              num_segments=1, num_crop=1, root='~/.mxnet/models',
                              ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=152,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet152_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                        use_tsn=False, partial_bn=False,
                        num_segments=1, num_crop=1, root='~/.mxnet/models',
                        ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=50,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.9,
                               init_std=0.001)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet50_v1b_ucf101',
                                             tag=pretrained, root=root))
        from ...data import UCF101Attr
        attrib = UCF101Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_hmdb51(nclass=51, pretrained=False, pretrained_base=True,
                        use_tsn=False, partial_bn=False,
                        num_segments=1, num_crop=1, root='~/.mxnet/models',
                        ctx=mx.cpu(), **kwargs):
    model = ActionRecResNetV1b(depth=50,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.9,
                               init_std=0.001)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet50_v1b_hmdb51',
                                             tag=pretrained, root=root))
        from ...data import HMDB51Attr
        attrib = HMDB51Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_custom(nclass=400, pretrained=False, pretrained_base=True,
                        use_tsn=False, partial_bn=False,
                        num_segments=1, num_crop=1, root='~/.mxnet/models',
                        ctx=mx.cpu(), use_kinetics_pretrain=True, **kwargs):
    model = ActionRecResNetV1b(depth=50,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if use_kinetics_pretrain and not pretrained:
        from gluoncv.model_zoo import get_model
        kinetics_model = get_model('resnet50_v1b_kinetics400', nclass=400, pretrained=True)
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
