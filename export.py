import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import set_logging, check_img_size
# from models.modal_ensemble_model_uva import ModalEnseModel
from utils.torch_utils import select_device
from torch.autograd import Variable

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

# from models.repvgg import C3RepVGG
# from gcn_lib.torch_vertex  import Grapher

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

_onnx_opset_version = 1


def register_custom_op():
    """
    This function registers symbolic functions for
    custom ops that are implemented as part of ONNX Runtime
    """

    # Symbolic definition
    def grid_sample(g, input, grid, mode, padding_mode, align_corners):
        # mode
        #   'bilinear'      : onnx::Constant[value={0}]
        #   'nearest'       : onnx::Constant[value={1}]
        #   'bicubic'       : onnx::Constant[value={2}]
        # padding_mode
        #   'zeros'         : onnx::Constant[value={0}]
        #   'border'        : onnx::Constant[value={1}]
        #   'reflection'    : onnx::Constant[value={2}]
        mode = sym_help._maybe_get_const(mode, "i")
        padding_mode = sym_help._maybe_get_const(padding_mode, "i")
        mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
        padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
        align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

        # From opset v13 onward, the output shape can be specified with
        # (N, C, H, W) (N, H_out, W_out, 2) => (N, C, H_out, W_out)
        # input_shape = input.type().sizes()
        # gird_shape = grid.type().sizes()
        # output_shape = input_shape[:2] + gird_shape[1:3]
        # g.op(...).setType(input.type().with_sizes(output_shape))

        return g.op("com.microsoft::GridSample", input, grid,
                    mode_s=mode_str,
                    padding_mode_s=padding_mode_str,
                    align_corners_i=align_corners)

    def inverse(g, self):
        return g.op("com.microsoft::Inverse", self).setType(self.type())

    def gelu(g, self):
        return g.op("com.microsoft::Gelu", self).setType(self.type())

    def triu(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=1).setType(self.type())

    def tril(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=0).setType(self.type())

    # Op Registration
    register_custom_op_symbolic('::grid_sampler', grid_sample, _onnx_opset_version)
    register_custom_op_symbolic('::inverse', inverse, _onnx_opset_version)
    register_custom_op_symbolic('::gelu', gelu, _onnx_opset_version)
    register_custom_op_symbolic('::triu', triu, _onnx_opset_version)
    register_custom_op_symbolic('::tril', tril, _onnx_opset_version)


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # print("ch", ch)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR, ]:

            if m is Focus:
                c1, c2 = 3, args[0]
                # print("focus c2", c2)
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR,]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Add:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            # print("Add2 arg", args[0])
            args = [c2, args[1]]
        elif m is GPT:
            c2 = ch[f[0]]
            args = [c2]
        elif m is MAM2:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        # if i == 4:
        #     ch = []
        ch.append(c2)
    # print(layers)
    return nn.Sequential(*layers), sorted(save)


class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict
            # print("YAML")
            # print(self.yaml)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # print(self.model)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # # Build strides, anchors
        m = self.model[-1]  # Detect()
        # print(m)

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # print("1, ch, s, s", 1, ch, s, s)
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            # print("m.stride", m.stride)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            # self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')


    def forward(self, x, x2, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, x2, profile)  # single-scale inference, train


    def forward_once(self, x, x2, profile=False):
        """

        :param x:          RGB Inputs
        :param x2:         IR  Inputs
        :param profile:
        :return:
        """
        y, dt = [], []  # outputs
        i = 0
        for m in self.model:
            # print("Moudle", i)
            if m.f != -1:  # if not from previous layer
                if m.f != -4:
                    # print(m)
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # print(len(y))
            i+=1

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module

        if isinstance(m, Detect):
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(
                    cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        elif isinstance(m, Decoupled_Detect):
            for mi, s in zip(m.m_conf, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            for mi, s in zip(m.m_cls, m.stride):  # from
                b = mi[-1].bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='models/transformer/yolov5l_fusion_add_FLIR_aligned.yaml',
                        help='model.yaml')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='runs/train/exp23/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    device = select_device('cpu')

    # Load PyTorch model
    # model = Model(opt.cfg)
    # model.load_weights(opt.weights, device)
    model = attempt_load(opt.weights, map_location=device)
    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection
    lwir = torch.zeros(opt.batch_size, 3, *opt.img_size)

    # img = img.to(device, non_blocking=True)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # lwir = lwir.to(device, non_blocking=True)
    lwir /= 255.0  # 0 - 255 to 0.0 - 1.0
    y = model(img, lwir, augment=False) # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        # torch.onnx.export(model, (img, lwir), f, verbose=True, opset_version=12, input_names=['img', 'lwir'],
        #                   output_names=['classes', 'boxes'],operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

        # model = UNet(3, 2)  # 自己定义的网络模型
        # model.load_state_dict(torch.load("best_weights.pth"))  # 保存的训练模型
        # model.eval()  # 切换到eval（）
        # example = torch.rand(1, 3, 320, 480)  # 生成一个随机输入维度的输入
        traced_script_module = torch.jit.trace(model, (img, lwir))
        traced_script_module.save("/home/xue/双模态对比实验/multispectral-object-detection-main/model.pt")

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    # print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))