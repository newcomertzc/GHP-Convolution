import torch
import numpy as np
import cv2 as cv
import torch.nn as nn
import torch.optim as torch_optim

from glob import glob
from tqdm import tqdm
from sklearn import metrics as sklearn_metrics
from torchvision import models as torchvision_models

from core import IMDNetwork
from core import backbone
from core import preprocessing
from core import convnext
from core import bayarcnn
from core import optim
from core.base import *
from core.data import ClfDataset
from core.data.functional import get_clf_dataloader
from core.common_types import *
from core.utils import *
from core.transforms import *


def main(args):
    set_seed(args.seed)
    keep_dir_valid(args.save_dir)
    use_deterministic_algorithms(args.use_deterministic_algorithms)
    device = torch.device(args.device)
    print(args)
    
    # get transform
    quality_values, quality_probs, subsampling_probs = get_JPEG_stat(args.data_stat)
    toJPEG = RandomJPEGCompress(quality_values, quality_probs, subsampling_probs)
    valid_input_type = {'green', 'gray', 'green'}
    if args.input_type not in valid_input_type:
        raise ValueError(f"Invalid input_type {args.input_type}. Only \"green\", \"gray\" and"
                         f"\"rgb\" are supported.")
    else:
        input_transform = get_input_transform(args.input_type)
    if args.patches_per_image == 1:
        toPatches = RandomPatch(args.patch_size, args.patch_stride)
    else:
        toPatches = RandomNPatches(args.patch_size, args.patch_stride, args.patches_per_image)
    transform, after_transform = get_default_transform(input_transform, toPatches, toJPEG)
    transform_val = Compose([PILToArray(), after_transform])
    
    # get dataset and dataloader
    paths = sorted(glob(path_join(args.data_path, '*')))
    dataloader = FunctionExecutor(
        func=get_clf_dataloader,
        kwargs={
            'paths': paths,
            'transform': transform,
            'after_transform': after_transform,
            'batch_size': args.batch_size,
            'single_patch': (args.patches_per_image == 1),
            'shuffle': True
        }
    )
    dataset_val = ClfDataset(args.data_val_path, transform_val)
    dataloader_val = DataLoader(dataset_val, args.batch_size)
    
    # get model
    in_channels = 1 if args.input_type != 'rgb' else 3
    num_classes = 10 # TODO customize num_classes
    model_func = get_model_func(args.backbone_func)
    
    if 'BayarCNN' in args.backbone:
        if args.input_type == 'rgb':
            raise ValueError(f"BayarCNN expect an input with 1 channels, but got input_type = {args.input_type} "
                             f"which means 3 channels.")
        backbone_kwargs = {
            'num_classes': num_classes
        }
        if 'GHP' in args.backbone:
            backbone_kwargs['penalty'] = args.preproc_reg
        
        model = IMDNetwork.build_with_kwargs(
            backbone_func = bayarcnn.__dict__[args.backbone],
            backbone_kwargs = backbone_kwargs
        )
    else:
        if args.preproc is None:
            model = IMDNetwork.build_with_kwargs(
                backbone_func = backbone.__dict__[args.backbone],
                backbone_kwargs = {
                    'in_channels': in_channels,
                    'num_classes': num_classes,
                    'model_func': model_func
                }
            )
        else:
            preprocessing_kwargs = {
                'in_channels': in_channels,
                'out_channels': args.preproc_width,
            }
            if 'GHP' in args.preproc:
                preprocessing_kwargs['penalty'] = args.preproc_reg
            
            model = IMDNetwork.build_with_kwargs(
                backbone_func = backbone.__dict__[args.backbone],
                backbone_kwargs = {
                    'in_channels': args.preproc_width,
                    'num_classes': num_classes,
                    'model_func': model_func
                },
                preprocessing_func = preprocessing.__dict__[args.preproc],
                preprocessing_kwargs = preprocessing_kwargs
            )
    
    print('***' * 12)
    print(f"name: {model.name}")
    if args.show_model:
        show_model(model, (in_channels, args.patch_size, args.patch_size))
    model.to(device)
    
    # get optimizer and loss function
    optimizer = get_opt_class(args.opt)(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss() if num_classes != 1 else nn.BCEWithLogitsLoss()
    
    # get metrics keeper 
    cmatrix_shape = num_classes if num_classes != 1 else 2
    
    metric_keeper = MetricKeeper()
    metric_keeper.add_metric('running_loss', [0.0, 0])
    metric_keeper.add_metric('running_val_loss', [0.0, 0])
    metric_keeper.add_metric(
        'train_cmatrix', np.zeros(shape=(cmatrix_shape, cmatrix_shape), dtype=np.int64))
    metric_keeper.add_metric(
        'val_cmatrix', np.zeros(shape=(cmatrix_shape, cmatrix_shape), dtype=np.int64))
    
    # load checkpoint    
    if args.checkpoint:
        print('load checkpoint...')
        start_epoch = load_checkpoint(model, optimizer, metric_keeper, args.checkpoint)
    else:
        start_epoch = 1
        metric_keeper.add_metric('loss_dict', {})
        metric_keeper.add_metric('val_loss_dict', {})
        metric_keeper.add_metric('train_cmatrix_dict', {})
        metric_keeper.add_metric('val_cmatrix_dict', {})
    
    # only test
    if args.test_only:
        print('***' * 12)
        print('start evaluating: ')
        evaluate(model, criterion, dataloader_val, device, -1, num_classes, metric_keeper)
        return
    
    print('***' * 12)
    print('start training: ')
    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(
            model, criterion, optimizer, dataloader, device, 
            epoch, num_classes, metric_keeper, args.print_freq)
        metric_keeper.reset_metric('running_loss')
        metric_keeper.reset_metric('train_cmatrix')
        
        if epoch % args.save_freq == 0:
            print('start evaluating: ')
            evaluate(model, criterion, dataloader_val, device, epoch, num_classes, metric_keeper)
            
            checkpoint_name = (
                f"{model.name}_{args.input_type}_epoch_{epoch:03d}_"
                f"valAcc_{calc_accuracy(metric_keeper.val_cmatrix) * 100:05.2f}_"
                f"valLoss_{metric_keeper.running_val_loss[0] / metric_keeper.running_val_loss[1]:.4f}.pt")
            save_checkpoint(model, optimizer, epoch, metric_keeper, args.save_dir, checkpoint_name)
            
            metric_keeper.reset_metric('running_val_loss')
            metric_keeper.reset_metric('val_cmatrix')
            print('start training: ')


def get_opt_class(opt: str) -> Optimizer:
    opt_classes = {}
    opt_classes.update(torch_optim.__dict__)
    opt_classes.update(optim.__dict__)
    
    return opt_classes[opt]


def get_model_func(model_func: str) -> Callable:
    model_funcs = {}
    model_funcs.update(torchvision_models.__dict__)
    model_funcs.update(backbone.__dict__)
    model_funcs.update(convnext.__dict__)
    
    return model_funcs[model_func]


def get_default_transform(input_transform: tuple, toPatches: Callable, toJPEG: Callable) -> tuple:
    input_convert, input_normalize = input_transform
    
    transform = Compose([
        RGBAToRGB(),
        RandomTransform([
            Compose([toPatches]),
            Compose([
                RandomTransform([
                    RandomResize_discrete(interpolation=cv.INTER_NEAREST),
                    RandomRotate_discrete(interpolation=cv.INTER_NEAREST),
                ]), toJPEG, toPatches,
            ]),
            Compose([
                RandomTransform([
                    RandomResize_discrete(interpolation=cv.INTER_LINEAR),
                    RandomRotate_discrete(interpolation=cv.INTER_LINEAR),
                ]), toJPEG, toPatches,
            ]),
            Compose([
                RandomTransform([
                    RandomResize_discrete(interpolation=cv.INTER_CUBIC),
                    RandomRotate_discrete(interpolation=cv.INTER_CUBIC),
                ]), toJPEG, toPatches,
            ]),
            Compose([RandomMedianFilter(), toJPEG, toPatches]),
            Compose([RandomBoxFilter(), toJPEG, toPatches]),
            Compose([RandomGaussianFilter_discrete(), toJPEG, toPatches]),
            Compose([RandomAWGN_discrete(), toJPEG, toPatches]),
            Compose([PoissonNoise(), toJPEG, toPatches]),
            Compose([RandomImpulseNoise_discrete(), toJPEG, toPatches]),
        ],
            return_label=True),
    ])
    after_transform = Compose([
        input_convert,
        ToContiguousArray(),
        ToTensor(),
        input_normalize])
    
    return transform, after_transform

    
def train_one_epoch(
    model: BaseModule, 
    criterion: Callable, 
    optimizer: Optimizer, 
    dataloader: FunctionExecutor,
    device: torch.device,
    epoch: int,
    num_classes: int,
    metric_keeper: MetricKeeper,
    print_freq: int = 10
) -> None:
    model.train()
    header = f"  Epoch: [{epoch}] "
    
    input_dtype = torch.float32
    label_dtype = torch.int64 if num_classes != 1 else torch.float32
    cmatrix_shape = num_classes if num_classes != 1 else 2
    
    dataloader = dataloader.execute()
    for iter_num, data in tqdm(enumerate(dataloader)):
        inputs, labels = data
        inputs = inputs.to(input_dtype).to(device)
        labels = labels.to(label_dtype).to(device)
        
        optimizer.zero_grad()
        out = model(inputs)
        if num_classes == 1:
            out = out.squeeze(dim=-1)
        loss = criterion(out, labels)
        
        reg_loss = model.calc_reg_loss()
        (loss + reg_loss).backward()
        
        optimizer.step()
        metric_keeper.running_loss[0] += loss.item()
        metric_keeper.running_loss[1] += 1
        
        y_true = labels.cpu()
        if num_classes == 1:
            y_pred = predict_binary(out)
        else:
            y_pred = predict_multiclass(out)
        cmatrix = sklearn_metrics.confusion_matrix(
            y_true, y_pred, labels=np.arange(cmatrix_shape))
        metric_keeper.train_cmatrix += cmatrix
        
        if (iter_num + 1) % print_freq == 0:
            print(header + 
                  f" lr: {optimizer.param_groups[0]['lr']} "
                  f" acc: {calc_accuracy(metric_keeper.train_cmatrix) * 100:05.2f} "
                  f" loss: {metric_keeper.running_loss[0] / metric_keeper.running_loss[1]:.4f} ", end='')
            if reg_loss != 0:
                print(f" reg loss: {reg_loss:.4f} ")
            else:
                print()
                
    metric_keeper.loss_dict[epoch] = metric_keeper.running_loss[0] / metric_keeper.running_loss[1]
    metric_keeper.train_cmatrix_dict[epoch] = metric_keeper.train_cmatrix
    
    del dataloader
                

def evaluate(
    model: BaseModule, 
    criterion: Callable, 
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    num_classes: int,
    metric_keeper: MetricKeeper
) -> None:
    model.eval()
    
    input_dtype = torch.float32
    label_dtype = torch.int64 if num_classes != 1 else torch.float32
    cmatrix_shape = num_classes if num_classes != 1 else 2
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, labels = data
            inputs = inputs.to(input_dtype).to(device)
            labels = labels.to(label_dtype).to(device)
            
            out = model(inputs)
            if num_classes == 1:
                out = out.squeeze(dim=-1)
            loss = criterion(out, labels)
            
            metric_keeper.running_val_loss[0] += loss.item()
            metric_keeper.running_val_loss[1] += 1
            
            y_true = labels.cpu()
            if num_classes == 1:
                y_pred = predict_binary(out)
            else:
                y_pred = predict_multiclass(out)
            metric_keeper.val_cmatrix += sklearn_metrics.confusion_matrix(
                y_true, y_pred, labels=np.arange(cmatrix_shape))
                
    metric_keeper.val_loss_dict[epoch] = (
        metric_keeper.running_val_loss[0] / metric_keeper.running_val_loss[1])
    metric_keeper.val_cmatrix_dict[epoch] = metric_keeper.val_cmatrix
    print(f" acc: {calc_accuracy(metric_keeper.val_cmatrix) * 100:05.2f} "
          f" loss: {metric_keeper.running_val_loss[0] / metric_keeper.running_val_loss[1]:.4f} ")
            

def save_checkpoint(
    model: BaseModule, 
    optimizer: Optimizer, 
    epoch: int, 
    metric_keeper: MetricKeeper, 
    save_dir: str,
    checkpoint_name: Optional[str] = None,
) -> None:
    checkpoint = {
        'model': model.state_dict(),
        'kwargs': model.kwargs,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': metric_keeper.loss_dict,
        'val_loss': metric_keeper.val_loss_dict,
        'train_cmatrix': metric_keeper.train_cmatrix_dict,
        'val_cmatrix': metric_keeper.val_cmatrix_dict
    }
    
    if checkpoint_name is None:
        checkpoint_name = f"{model.name}_{epoch}.pt"
    torch.save(checkpoint, path_join(save_dir, checkpoint_name))
    

def load_checkpoint(
    model: BaseModule, 
    optimizer: Optimizer, 
    metric_keeper: MetricKeeper,
    checkpoint: str
) -> int:
    """Load checkpoint.

    Returns:
        int: start_epoch
    """
    checkpoint = torch.load(checkpoint, 'cpu')
    print_inconsistent_kwargs(model.kwargs, checkpoint['kwargs'])
    
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    if checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    metric_keeper.add_metric('loss_dict', checkpoint['loss'])
    metric_keeper.add_metric('val_loss_dict', checkpoint['val_loss'])
    metric_keeper.add_metric('train_cmatrix_dict', checkpoint['train_cmatrix'])
    metric_keeper.add_metric('val_cmatrix_dict', checkpoint['val_cmatrix'])
    
    return start_epoch


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description='IMD Classification Training', add_help=add_help)

    parser.add_argument('--data-path', default='dataset/imagenet-train-40c-50ipc/', type=str, help='training set path')
    parser.add_argument('--data-val-path', default='dataset/imagenet-val-10800/', type=str, help='validation set path')
    parser.add_argument('--data-stat', default='statistics/imagenet_statistics.pkl', type=str, help='path of dataset compression statistics file')
    parser.add_argument(
        '-i', '--input-type', default='green', type=str, help='"green", "gray" or "rgb"'
    )
    parser.add_argument('--patches-per-image', default=4, type=int, help='maximum number of patches per image')
    parser.add_argument('--patch-size', default=224, type=int, help='patch size')
    parser.add_argument('--patch-stride', default=160, type=int, help='(maximum) patch stride')
    # parser.add_argument(
    #     '-n', '--num-classes', default=10, type=int, help='total number of classes'
    # )
    parser.add_argument('--backbone', default='ResNet', type=str, help='name of backbone network')
    parser.add_argument('--backbone-func', default='resnet50', type=str, help='name of backbone function')
    parser.add_argument('--preproc', default=None, type=str, help='name of preprocessing module (Default: None)')
    parser.add_argument('--preproc-width', default=12, type=int, help='output channels of preprocessing module')
    parser.add_argument('--preproc-reg', default='L1', type=str, help='"L1" or "L2"')
    parser.add_argument('--device', default='cuda', type=str, help='device (Use cuda or cpu, Default: cuda)')
    parser.add_argument(
        '-b', '--batch-size', default=32, type=int, help='images per batch'
    )
    parser.add_argument('--epochs', default=800, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--opt', default='RAdam', type=str, help='optimizer')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    # parser.add_argument(
    #     '--label-smoothing', default=0.0, type=float, help='label smoothing (default: 0.0)', dest='label_smoothing'
    # )
    parser.add_argument('--show-model', action='store_true', help='Show model information.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--save-freq', default=20, type=int, help='save frequency')
    parser.add_argument('--save-dir', default='saved_models/', type=str, help='path to save checkpoint')
    parser.add_argument('--checkpoint', default=None, type=str, help='path of the checkpoint to load')
    # parser.add_argument(
    #     "--pretrained",
    #     dest="pretrained",
    #     help="Use pre-trained models from the modelzoo",
    #     action="store_true",
    # )
    parser.add_argument(
        '--test-only',
        dest='test_only',
        help='Only test the model',
        action='store_true',
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument('--seed', default=2021, type=int, help='random seed')

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)