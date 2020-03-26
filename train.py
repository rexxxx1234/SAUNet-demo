# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
import math
# Numerical libs
import torch
import torch.nn as nn
import torch.utils.data as data
from data.augmentations import Compose, RandomSizedCrop, AdjustContrast, AdjustBrightness, RandomVerticallyFlip, RandomHorizontallyFlip, RandomRotate, PaddingCenterCrop
# Our libs
from data.dataloader import LungData
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices, accuracy, intersectionAndUnion
from lib.nn import UserScatteredDataParallel, async_copy_to,  user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
from lib.utils import as_numpy
import numpy as np
from loss import DualLoss
import cv2


import pydicom as dicom
from scipy.ndimage.morphology import distance_transform_edt

def visualize_result(data, seg, pred, args):
    img = data[0][0]

    #normalize image to [0, 1] first.
    img = (img - img.min())/(img.max()-img.min())
    img = (img * 255).astype(np.uint8) #Then scale it up to [0, 255] to get the final image.
    pred_img = (pred * 51).astype(np.uint8)
    seg = (seg*51).astype(np.uint8)
    #print(img.shape, pred_img.shape)
    #heat = get_heatmap(LRP)
    im_vis = np.concatenate((img, seg, pred_img), axis=1).astype(np.uint8)
    img_name = str(random.randrange(10000, 99999)) + '.png'
    cv2.imwrite(os.path.join(args.result,
                img_name), im_vis)

def visualize_dcm(data, path):
    img = data[0][0]

    #normalize image to [0, 1] first.
    img = (img - img.min())/(img.max()-img.min())
    img = (img * 255).astype(np.uint8) #Then scale it up to [0, 255] to get the final image.
    print(img.shape)
    #print(img.shape, pred_img.shape)
    #heat = get_heatmap(LRP)
    img_name = 'display.png'
    cv2.imwrite(os.path.join(path,
                img_name), img)

def visualize_pred(pred, path):
    pred_img = (pred * 51).astype(np.uint8)
    #print(img.shape, pred_img.shape)
    #heat = get_heatmap(LRP)
    img_name = 'display.png'
    cv2.imwrite(os.path.join(path,
                img_name), pred_img)



def eval(loader_val, segmentation_module, args, crit):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    loss_meter = AverageMeter()

    segmentation_module.eval()
    for batch_data in loader_val:
        batch_data = batch_data[0]
        
        seg_label = as_numpy(batch_data["mask"][0])
        torch.cuda.synchronize()
        batch_data["image"] = batch_data["image"].unsqueeze(0).cuda()
        print(batch_data["image"].shape)

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)
            print("the score:", scores)
            feed_dict = batch_data.copy()
            

            # forward pass
            scores_tmp, loss = segmentation_module(feed_dict, epoch=0, segSize=segSize)
            scores = scores + scores_tmp
            print("the new score:", scores)
            loss_meter.update(loss)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            print("pred shape:", pred.shape)
            
            visualize_result(batch_data["image"].cpu().numpy(), seg_label, pred, args)

        torch.cuda.synchronize()
        # calculate accuracy
        intersection, union = intersectionAndUnion(pred, seg_label, args.num_class)
        intersection_meter.update(intersection)
        union_meter.update(union)
    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        if i >= 1:
            print('class [{}], IoU: {:.4f}'.format(i, _iou))
    print('loss: {:.4f}'.format(loss_meter.average()))
    return iou[1:], loss_meter.average()

# train one epoch
def train(segmentation_module, loader_train, optimizers, history, epoch, args):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_j1 = AverageMeter()
    ave_j2 = AverageMeter()
    ave_j3 = AverageMeter()
    ave_j4 = AverageMeter()
    ave_j5 = AverageMeter()

    segmentation_module.train(not args.fix_bn)

    # main loop
    tic = time.time()
    iter_count = 0

    if epoch == args.start_epoch and args.start_epoch > 1:
        scale_running_lr = ((1. - float(epoch-1) / (args.num_epoch)) ** args.lr_pow)
        args.running_lr_encoder = args.lr_encoder * scale_running_lr
        for param_group in optimizers[0].param_groups:
            param_group['lr'] = args.running_lr_encoder

    for batch_data in loader_train:
        data_time.update(time.time() - tic)
        batch_data["image"] = batch_data["image"].cuda()
        segmentation_module.zero_grad()
        # forward pass
        loss, acc = segmentation_module(batch_data, epoch)
        loss = loss.mean()

        jaccard = acc[1]
        for j in jaccard:
            j = j.float().mean()
        acc = acc[0].float().mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        iter_count += args.batch_size_per_gpu

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        ave_j1.update(jaccard[0].data.item()*100)
        ave_j2.update(jaccard[1].data.item()*100)
        ave_j3.update(jaccard[2].data.item()*100)
        ave_j4.update(jaccard[3].data.item()*100)
        ave_j5.update(jaccard[4].data.item()*100)

        if iter_count % (args.batch_size_per_gpu*10) == 0:
            # calculate accuracy, and display
            if args.unet==False:
                print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                        'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                        'Accuracy: {:4.2f}, Loss: {:.6f}'
                        .format(epoch, i, args.epoch_iters,
                        batch_time.average(), data_time.average(),
                        args.running_lr_encoder, args.running_lr_decoder,
                        ave_acc.average(), ave_total_loss.average()))
            else:
                print('Epoch: [{}/{}], Iter: [{}], Time: {:.2f}, Data: {:.2f},'
                        ' lr_unet: {:.6f}, Accuracy: {:4.2f}, Jaccard: [{:4.2f},{:4.2f},{:4.2f},{:4.2f},{:4.2f}] '
                        'Loss: {:.6f}'
                        .format(epoch, args.max_iters, iter_count,
                            batch_time.average(), data_time.average(),
                            args.running_lr_encoder, ave_acc.average(),
                            ave_j1.average(), ave_j2.average(),
                            ave_j3.average(), ave_j4.average(),
                            ave_j5.average(), ave_total_loss.average()))

    #Average jaccard across classes.
    j_avg = (ave_j1.average() + ave_j2.average() + ave_j3.average() + ave_j4.average() + ave_j5.average())/5

    #Update the training history
    history['train']['epoch'].append(epoch)
    history['train']['loss'].append(loss.data.item())
    history['train']['acc'].append(acc.data.item())
    history['train']['jaccard'].append(j_avg)
    # adjust learning rate
    adjust_learning_rate(optimizers, epoch, args)


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    (unet, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))

    dict_unet = unet.state_dict()
    torch.save(dict_unet,
                '{}/unet_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (unet, crit) = nets
    if args.optimizer.lower() == 'sgd':
        optimizer_unet = torch.optim.SGD(
            group_weight(unet),
            lr=args.lr_encoder,
            momentum=args.beta1,
            weight_decay=args.weight_decay,
            nesterov=False)
    elif args.optimizer.lower() == 'adam':
        optimizer_unet = torch.optim.Adam(
            group_weight(unet),
            lr = args.lr_encoder,
            betas=(0.9, 0.999))
    return [optimizer_unet]


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = 0.5*(1+math.cos(3.14159*(cur_iter)/args.num_epoch))
    args.running_lr_encoder = args.lr_encoder * scale_running_lr

    optimizer_unet = optimizers[0]
    for param_group in optimizer_unet.param_groups:
        param_group['lr'] = args.running_lr_encoder


def mask_to_onehot(mask, num_classes=5):
    _mask = [mask == i for i in range(1, num_classes+1)]
    _mask = [np.expand_dims(x, 0) for x in _mask]
    return np.concatenate(_mask, 0)

def mask_to_edges(mask):
    _edge = mask
    _edge = mask_to_onehot(_edge)
    _edge = onehot_to_binary_edges(_edge)
    return torch.from_numpy(_edge).float()

def onehot_to_binary_edges(mask, radius=2, num_classes=3):
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap



def preprocess(inp):
    slice1 = dicom.read_file(inp)
    img = slice1.pixel_array
    seg = img.copy() # Since the model needs seg as input, create a dumb seg here just to match the format

    img -= img.min()
    augmentations = Compose([PaddingCenterCrop(256)])
    img, seg = augmentations(img.astype(np.uint32), seg.astype(np.uint8))
    
    mu = img.mean()
    sigma = img.std()
    img = (img - mu) / (sigma+1e-10)

    if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)

    img = torch.from_numpy(img).float()
    mask = mask_to_edges(seg)
    seg = torch.from_numpy(seg).long()

    data_dict = {
        "image": img,
        "mask": (seg, mask),
    }

    print("finish preprocess")

    return data_dict

def main(args):
    # Network Builders
    builder = ModelBuilder()

    unet = builder.build_unet(num_class=args.num_class,
        arch=args.unet_arch,
        weights=args.weights_unet)

    print("Froze the following layers: ")
    for name, p in unet.named_parameters():
        if p.requires_grad == False:
            print(name)
    print()

    crit = DualLoss(mode="train")

    segmentation_module = SegmentationModule(crit, unet)

    test_augs = Compose([PaddingCenterCrop(256)])
    
    print("ready to load data")

    dataset_val = LungData( 
            root=args.data_root,
            split='test',
            k_split=args.k_split,
            augmentations=test_augs)

    
    loader_val = data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    print(len(loader_val))

    # load nets into gpu
    if len(args.gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=args.gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit) if args.unet == False else (unet, crit)
    optimizers = create_optimizers(nets, args)

    '''
    # Start the webapp: user update a dcm file, output the predicted segmentation pic of it
    inp = gradio.inputs.DcmUpload(preprocessing_fn=preprocess)
    #inp = gradio.inputs.ImageUpload(preprocessing_fn=preprocess)
    io = gradio.Interface(inputs=inp, outputs="image", model_type="lung_seg", model=segmentation_module, args=args)
    io.launch(validate=False)
    '''

    iou, loss = eval(loader_val, segmentation_module, args, crit)
    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    #DATA_ROOT = os.getenv('DATA_ROOT', '/home/rexma/Desktop/MRI_Images/LCTSC')
    DATA_ROOT = os.getenv('DATA_ROOT', '/home/rexma/demo/LCTSC')
    DATASET_NAME = "Lung"

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='200317',
                        help="a name for identifying the model")
    parser.add_argument('--unet', default=True,
                        help="use unet?")
    parser.add_argument('--unet_arch', default='saunet',
                        help="UNet architecture")
    parser.add_argument('--weights_unet', default='/home/rexma/Desktop/JesseSun/lungseg/ckpt/200316-saunet-ngpus1-batchSize10-LR_unet0.0003-epoch180/unet_epoch_45.pth',
                        help="weights to finetune unet")

    # Path related arguments
    parser.add_argument('--data-root', type=str, default=DATA_ROOT)
    parser.add_argument('--result', default='./result')

    # optimization related arguments
    parser.add_argument('--gpus', default='0',
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=120, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=160, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='Adam', help='optimizer')
    parser.add_argument('--lr_encoder', default=0.0005, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', action='store_true',
                        help='fix bn params')

    # Data related argument
    parser.add_argument('--num_class', default=6, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loading workers')
    parser.add_argument('--dataset-name', type=str, default="Lung")
    parser.add_argument('--k_split', default=1)

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')

    parser.add_argument('--optimizer', default='sgd')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if args.optimizer.lower() in ['sgd', 'adam', 'radam']:
        # Parse gpu ids
        all_gpus = parse_devices(args.gpus)
        all_gpus = [x.replace('gpu', '') for x in all_gpus]
        args.gpus = [int(x) for x in all_gpus]
        num_gpus = len(args.gpus)
        args.batch_size = num_gpus * args.batch_size_per_gpu
        args.gpu = 0

        args.max_iters = args.num_epoch
        args.running_lr_encoder = args.lr_encoder

        # Model ID
        if args.unet ==False:
            args.id += '-' + args.arch_encoder
            args.id += '-' + args.arch_decoder
        else:
            args.id += '-' + str(args.unet_arch)

        args.id += '-ngpus' + str(num_gpus)
        args.id += '-batchSize' + str(args.batch_size)

        args.id += '-LR_unet' + str(args.lr_encoder)

        args.id += '-epoch' + str(args.num_epoch)

        print('Model ID: {}'.format(args.id))

        args.ckpt = os.path.join(args.ckpt, args.id)
        if not os.path.isdir(args.ckpt):
            os.makedirs(args.ckpt)

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        #main(args)

        ''''''
        #build model
        builder = ModelBuilder()

        unet = builder.build_unet(num_class=args.num_class,
            arch=args.unet_arch,
            weights=args.weights_unet)

        print("Froze the following layers: ")
        for name, p in unet.named_parameters():
            if p.requires_grad == False:
                print(name)
        print()

        crit = DualLoss(mode="train")

        segmentation_module = SegmentationModule(crit, unet)

        # load nets into gpu
        if len(args.gpus) > 1:
            segmentation_module = UserScatteredDataParallel(
                segmentation_module,
                device_ids=args.gpus)
            # For sync bn
            patch_replication_callback(segmentation_module)
        segmentation_module.cuda()

        print("ready to load data")

        import train
        import streamlit as st
        from os import listdir
        from os.path import isfile, join
        from PIL import Image
        test_path = '/home/rexma/demo/LCTSC/demo/'
        temp_path = '/home/rexma/demo/LCTSC/temp/'
        result_path = '/home/rexma/demo/LCTSC/result/'
        st.sidebar.title("About")
        st.sidebar.info(
            "This is a demo application written to help you understand Streamlit. The application identifies the animal in the picture. It was built using a Convolution Neural Network (CNN).")

        st.sidebar.title("Predict New Images")
        onlyfiles = [f for f in listdir(test_path) if isfile(join(test_path, f))]
        imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)

        

        uploaded_file = st.file_uploader("Choose an dicom file...", type="dcm")

        if uploaded_file is not None:
    
            #data_dict = preprocess(test_path + imageselect)
            data_dict = preprocess(uploaded_file)
            
            # visulizatio of original file 
            seg_label = as_numpy(data_dict["mask"][0])
            torch.cuda.synchronize()
            data_dict["image"] = data_dict["image"].unsqueeze(0).cuda()

            visualize_dcm(data_dict["image"].cpu().numpy(), temp_path)
            image = Image.open(temp_path + 'display.png')
            st.image(image, caption="Original dicom file!", use_column_width=True)

            
            # segmentation
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)
            feed_dict = data_dict.copy()

            scores_tmp, loss = segmentation_module(feed_dict, epoch=0, segSize=segSize)
            scores = scores + scores_tmp

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())


            visualize_pred(pred, result_path)
            image = Image.open(result_path + 'display.png')
            st.image(image, caption="After lung segmentation!", use_column_width=True)
        

    else:
        print("Invalid optimizer. Please try again with optimizer sgd, adam, or radam.")






