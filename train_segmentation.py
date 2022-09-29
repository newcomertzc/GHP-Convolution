import torch
import numpy as np
import cv2 as cv
import torch.nn as nn
import torch.optim as torch_optim

from tqdm import tqdm
from pycocotools.coco import COCO
from sklearn import metrics as sklearn_metrics

from core import IMDNetwork
from core import backbone
from core import preprocessing
from core import optim
from core.base import *
from core.data import SegDataset
from core.data.functional import get_seg_dataloader_coco, collate_fn_seg
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
    input_type = get_base_name(args.pretrained).split('_')[1]
    input_transform = get_input_transform(input_type)
    
    toPatches = AllPatches(args.patch_size, args.patch_stride)
    pre_transform, transform, after_transform = get_default_transform(
        input_transform, toPatches, toJPEG)
    transform_val = Compose([PILToArray(), after_transform])
    
    # get dataset and dataloader
    dataloader = FunctionExecutor(
        func = get_seg_dataloader_coco,
        kwargs = {
            'coco': COCO(args.coco_ann_path),
            'image_dir': args.coco_path,
            'pre_transform': pre_transform,
            'transform': transform,
            'after_transform': after_transform,
            'batch_size': args.batch_size,
            'single_patch': False,
            'shuffle': True
        }
    )
    dataset_val = SegDataset(args.data_val_path, transform_val, PILToArray())
    dataloader_val = DataLoader(dataset_val, args.batch_size, collate_fn_seg)
    
    # get model
    in_channels = 1 if input_type != 'rgb' else 3
    num_classes = 10 # TODO customize num_classes
    pretrained = torch.load(args.pretrained, 'cpu')
    kwargs = pretrained['kwargs']
    
    backbone_kwargs = {
        'in_channels': kwargs['backbone_kwargs']['in_channels'],
        'num_classes': kwargs['backbone_kwargs']['num_classes'],
        'backbone_kwargs': {
            'model_func': kwargs['backbone_kwargs']['model_func'],
        },
    }
    if 'ResNet' in args.backbone:
        backbone_kwargs['replace_stride_with_dilation'] = args.replace_stride_with_dilation
    
    model = IMDNetwork.build_with_kwargs(
        backbone_func = backbone.segmentation.__dict__[args.backbone],
        backbone_kwargs = backbone_kwargs,
        preprocessing_func = kwargs['preprocessing_func'],
        preprocessing_kwargs = kwargs['preprocessing_kwargs']
    )
        
    print('***' * 12)
    print(f"name: {model.name}")
    if args.show_model:
        show_model(model, (in_channels, args.patch_size, args.patch_size))
        
    if kwargs['preprocessing_func'] is not preprocessing.PreprocIdentity:
        print('load pretrained preprocessing:')
        print(' ' + str(model.preprocessing.load_state_dict(
            extract_weights(pretrained['model'], desired_layer_name='preprocessing.', target_layer_name=''))))
        freeze_weights(model.preprocessing)
    model.backbone.load_backbone_weights(pretrained['model'], print_result=True)
    model.backbone.freeze_backbone_weights()
        
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
        start_epoch = load_checkpoint(model, optimizer, metric_keeper, args.checkpoint, device)
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
                f"{model.name}_{input_type}_epoch_{epoch:03d}_"
                f"valMIoU_{calc_mIoU(metric_keeper.val_cmatrix) * 100:05.2f}_"
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


def get_model(model: str) -> Callable:
    models = {}
    models.update(backbone.segmentation.__dict__)
    
    return models[model]


def get_default_transform(input_transform: tuple, toPatches: Callable, toJPEG: Callable) -> tuple:
    input_convert, input_normalize = input_transform
    
    pre_transform = Compose([
        RGBAToRGB(),
        RandomTransform([
            Compose([
                RandomTransform([
                    RandomResize_discrete(interpolation=cv.INTER_NEAREST),
                    RandomRotate_discrete(interpolation=cv.INTER_NEAREST),]), 
                toJPEG,
            ]),
            Compose([
                RandomTransform([
                    RandomResize_discrete(interpolation=cv.INTER_LINEAR),
                    RandomRotate_discrete(interpolation=cv.INTER_LINEAR),]), 
                toJPEG,
            ]),
            Compose([
                RandomTransform([
                    RandomResize_discrete(interpolation=cv.INTER_CUBIC),
                    RandomRotate_discrete(interpolation=cv.INTER_CUBIC),]), 
                toJPEG, 
            ]),
            Compose([RandomMedianFilter(), toJPEG]),
            Compose([RandomBoxFilter(), toJPEG]),
            Compose([RandomGaussianFilter(), toJPEG]),
            Compose([RandomAWGN_discrete(), toJPEG]),
            Compose([PoissonNoise(), toJPEG]),
            Compose([RandomImpulseNoise_discrete(), toJPEG]),
        ],
            return_label=True),
    ])
    transform = Compose([
        toPatches,
        SelectPatches(),
    ])
    after_transform = Compose([
        input_convert,
        ToContiguousArray(),
        ToTensor(),
        input_normalize])
    
    return pre_transform, transform, after_transform
    
    
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
        inputs, labels = data['image'], data['mask']
        inputs = inputs.to(input_dtype).to(device)
        labels = labels.to(label_dtype).to(device)
        
        optimizer.zero_grad()
        res = model(inputs)
        
        out, aux = res.get('out'), res.get('aux')
        if num_classes == 1:
            out = out.squeeze(dim=1)
            aux = aux.squeeze(dim=1) if aux is not None else aux
        loss = criterion(out, labels)
        aux_loss = criterion(aux, labels) if aux is not None else 0.0
        
        # reg_loss = model.calc_reg_loss()
        # (loss + aux_loss + reg_loss).backward()
        (loss + aux_loss).backward()
        
        optimizer.step()
        metric_keeper.running_loss[0] += loss.item()
        metric_keeper.running_loss[1] += 1
        
        y_true = labels.cpu()
        if num_classes == 1:
            y_pred = predict_binary(out)
        else:
            y_pred = predict_multiclass(out)
        cmatrix = sklearn_metrics.confusion_matrix(
            y_true.ravel(), y_pred.ravel(), labels=np.arange(cmatrix_shape))
        metric_keeper.train_cmatrix += cmatrix
        
        if (iter_num + 1) % print_freq == 0:
            print(header + 
                  f" lr: {optimizer.param_groups[0]['lr']} "
                #   f" acc: {calc_accuracy(metric_keeper.train_cmatrix)*100:05.2f} "
                  f" mIoU: {calc_mIoU(metric_keeper.train_cmatrix)*100:05.2f} "
                  f" loss: {metric_keeper.running_loss[0] / metric_keeper.running_loss[1]:.4f} ", end='')
            if aux_loss != 0:
                print(f" aux loss: {aux_loss:.4f} ")
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
            inputs, labels = data['image'], data['mask']
            inputs = inputs.to(input_dtype).to(device)
            labels = labels.to(label_dtype).to(device)
            
            res = model(inputs)
            
            out = res.get('out')
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
                y_true.ravel(), y_pred.ravel(), labels=np.arange(cmatrix_shape))
                
    metric_keeper.val_loss_dict[epoch] = (
        metric_keeper.running_val_loss[0] / metric_keeper.running_val_loss[1])
    metric_keeper.val_cmatrix_dict[epoch] = metric_keeper.val_cmatrix
    print(# f" acc: {calc_accuracy(metric_keeper.val_cmatrix)*100:05.2f} "
          f" mIoU: {calc_mIoU(metric_keeper.val_cmatrix)*100:05.2f} "
          f" loss: {metric_keeper.running_val_loss[0] / metric_keeper.running_val_loss[1]:.4f}")
            

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
    checkpoint: str,
    device: torch.device
) -> int:
    """Load checkpoint.

    Returns:
        int: start_epoch
    """
    checkpoint = torch.load(checkpoint, device)
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

    parser = argparse.ArgumentParser(description='IMD Segmentation Training', add_help=add_help)

    parser.add_argument('--coco-path', default='dataset/coco/train2017/', type=str, help='COCO training set path')
    parser.add_argument('--coco-ann-path', default='dataset/coco/annotations/instances_train2017.json', type=str, help='COCO annotation file path')
    parser.add_argument('--data-val-path', default='dataset/coco-val-seg/', type=str, help='validation set path')
    parser.add_argument('--data-stat', default='stat/imagenet_stat.pkl', type=str, help='path of dataset compression statistics file')
    # parser.add_argument(
    #     '-i', '--input-type', default='green', type=str, help='"green", "gray" or "rgb"'
    # )
    parser.add_argument('--patch-size', default=224, type=int, help='patch size')
    parser.add_argument('--patch-stride', default=160, type=int, help='(maximum) patch stride')
    # parser.add_argument(
    #     '-n', '--num-classes', default=10, type=int, help='total number of classes'
    # )
    
    parser.add_argument('--backbone', default='DeepLabV3_ResNet', type=str, help='backbone network (default: DeepLabV3_ResNet)')
    parser.add_argument('--pretrained', default=None, type=str, help='path of pretrained checkpoint')
    parser.add_argument(
        '--replace-stride-with-dilation', 
        default=[False, True, True], 
        type=str_to_bool, 
        nargs='+', 
        help='Adjust the dilation of ResNet. (default: [False, True, True])'
    )
    parser.add_argument('--device', default='cuda', type=str, help='device (Use cuda or cpu, default: cuda)')
    parser.add_argument(
        '-b', '--batch-size', default=32, type=int, help='images per batch'
    )
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--opt', default='RAdam', type=str, help='optimizer')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    # parser.add_argument(
    #     '--label-smoothing', default=0.0, type=float, help='label smoothing (default: 0.0)', dest='label_smoothing'
    # )
    parser.add_argument('--show-model', action='store_true', help='Show model information.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--save-freq', default=1, type=int, help='save frequency')
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
