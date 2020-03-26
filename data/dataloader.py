import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pydicom as dicom
import os
from skimage.draw import polygon

import os, nibabel
import sys, getopt
import PIL
from PIL import Image
import imageio
import scipy.misc
import numpy as np
import glob
from torch.utils import data
import torch
import random
from .augmentations import Compose, RandomRotate, PaddingCenterCrop
from skimage import transform

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample

class LungData(data.Dataset):

    def __init__(self,
                 root,
                 split='train',
                 augmentations=None,
                 img_norm=True,
                 k=5,
                 k_split=1,
                 target_size=(256, 256)
                 ):
        self.target_size = target_size
        self.ROOT_PATH = root
        self.split = split
        self.k = k
        self.split_len = int(200/self.k)
        self.k_split = int(k_split)
        self.augmentations = augmentations
        self.TRAIN_IMG_PATH = os.path.join(root, 'train', 'img')
        self.TRAIN_SEG_PATH = os.path.join(root, 'train', 'seg')
        self.list = self.read_files()

    def read_files(self):
        d = []
        print("for sure here")
        print(self.split)
        if self.split == 'train' or self.split == 'val':
            print("hello1")
            data_path = self.ROOT_PATH+"/train"
        else:
            print("hello2")
            data_path = self.ROOT_PATH+"/test"

        tmp_list = [os.path.join(data_path, name)
                          for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]

        split_len = len(tmp_list)//self.k
        patient_list = []

        for i in range(len(tmp_list)):
            if i in range((self.k-1)*split_len, self.k*split_len) and self.split == 'val':
                patient_list.append(tmp_list[i])
            elif i not in range((self.k-1)*split_len, self.k*split_len) and self.split == 'train':
                patient_list.append(tmp_list[i])
            elif self.split == 'trainval':
                patient_list.append(tmp_list[i])

        if self.split == 'test':
            patient_list = tmp_list

        for patient in patient_list:
            for dirs, subdir, files in os.walk(patient):
                tmp = sorted(subdir, key=lambda x: len(os.listdir(os.path.join(dirs, x))))
                if len(tmp) == 2:
                    ordered_subdirs = tmp 
                    ordered_subdirs = [os.path.join(dirs, subdirname) for subdirname in ordered_subdirs]
            for subdirname in ordered_subdirs:
                if any('.dcm' in elem for elem in os.listdir(subdirname)) and len(os.listdir(subdirname))==1:
                    structure = dicom.read_file(os.path.join(subdirname, os.listdir(subdirname)[0]))
                    contours = self.read_structure(structure)
                elif any('.dcm' in elem for elem in os.listdir(subdirname)) and len(os.listdir(subdirname))>1:
                    dcms = [os.path.join(subdirname, dcm) for dcm in os.listdir(subdirname)]
                    
                    slice1 = dicom.read_file(dcms[0])
                    image1 = slice1.pixel_array
                    print("one:", image1.shape)

                    slices = [dicom.read_file(dcm) for dcm in dcms]
                    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
                    image = np.stack([s.pixel_array for s in slices], axis=-1)
                    label, colors = self.get_mask(contours, slices, image.shape)
                    image = image.transpose(2,0,1)
                    label = label.transpose(2,0,1)
                    for i in range(image.shape[0]):
                        d.append((image[i], label[i]))
                        print("two1:", image[i].shape)
                        print("two2:", label[i].shape)
                        if np.array_equal(image1 , image[i]):
                            print("same")
                        else:
                            print("not same")
                        break

        return d

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i): # i is index
        img, seg = self.list[i]
        img -= img.min()
        img, seg = self.augmentations(img.astype(np.uint32), seg.astype(np.uint8))
        
        if random.uniform(0, 1.0) <= 0.5 and self.split != 'test' and self.split != 'val':
            img = augment_gamma(img)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            seg = np.expand_dims(seg, axis=2)
            stacked = np.concatenate((img, seg), axis=2)
            red = self.random_elastic_deformation(stacked, alpha=500, sigma=20).transpose(2,0,1)
            img, seg = red[0], red[1]

        mu = img.mean()
        sigma = img.std()
        img = (img - mu) / (sigma+1e-10)
        
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)

        img = torch.from_numpy(img).float()
        mask = self.mask_to_edges(seg)
        seg = torch.from_numpy(seg).long()
  
        data_dict = {
            "image": img,
            "mask": (seg, mask),
        }

        return data_dict

    def _transform(self, img, mask):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        return img, mask

    def get_mask(self, contours, slices, image):
        z = [round(s.ImagePositionPatient[2],1) for s in slices] ##
        pos_r = slices[0].ImagePositionPatient[1]
        spacing_r = slices[0].PixelSpacing[1]
        pos_c = slices[0].ImagePositionPatient[0]
        spacing_c = slices[0].PixelSpacing[0]

        label = np.zeros(image, dtype=np.uint8)
        for con in contours:
            num = int(con['number'])
            for c in con['contours']:
                nodes = np.array(c).reshape((-1, 3)) #triplets describing points of contour
                assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                z_index = z.index(np.around(nodes[0, 2], 1))
                r = (nodes[:, 1] - pos_r) / spacing_r
                c = (nodes[:, 0] - pos_c) / spacing_c
                rr, cc = polygon(r, c)
                label[rr, cc, z_index] = num

            colors = tuple(np.array([con['color'] for con in contours]) / 255.0)
        return label, colors

    def read_structure(self, structure):
        contours = []
        for i in range(len(structure.ROIContourSequence)):
            contour = {}
            contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
            contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber ##
            contour['name'] = structure.StructureSetROISequence[i].ROIName
            assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
            contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
            contours.append(contour)
        return contours

    def mask_to_onehot(self, mask, num_classes=5):
        _mask = [mask == i for i in range(1, num_classes+1)]
        _mask = [np.expand_dims(x, 0) for x in _mask]
        return np.concatenate(_mask, 0)
    
    def mask_to_edges(self, mask):
        _edge = mask
        _edge = self.mask_to_onehot(_edge)
        _edge = self.onehot_to_binary_edges(_edge)
        return torch.from_numpy(_edge).float()

    def onehot_to_binary_edges(self, mask, radius=2, num_classes=3):
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

    def random_elastic_deformation(self, image, alpha, sigma, mode='nearest',
                                   random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
    ..  [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        height, width, channels = image.shape

        dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        indices = (np.repeat(np.ravel(x+dx), channels),
                np.repeat(np.ravel(y+dy), channels),
                np.tile(np.arange(channels), height*width))

        values = map_coordinates(image, indices, order=1, mode=mode)

        return values.reshape((height, width, channels))

class AC17_2DLoad():
    def __init__(self, dataset, split='train', deform=True):
        super(AC17_2DLoad, self).__init__()
        self.data = []
        self.split = split
        self.deform = deform

        for i in range(dataset.__len__()):
            d = dataset[i]
            for x in range(d["image"].shape[-1]):
                entry = {}
                entry["image"] = d["image"].permute(2,0,1)[x]
                entry["mask"] = d["mask"].permute(2,0,1)[x]
                entry["name"] = d["name"] + "_z"+str(x)
                self.data.append(entry)

    def __getitem__(self, i):
        #return self.data[i]
        img = self.data[i]["image"]
        seg = self.data[i]["mask"]

        if self.split == 'train': #50% chance of deformation
            img = img.double().numpy()
            seg = seg.double().numpy()

            if random.uniform(0, 1.0) <= 0.5 and self.deform==True:
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                seg = np.expand_dims(seg, axis=2)
                stacked = np.concatenate((img, seg), axis=2)
                red = self.random_elastic_deformation(stacked, alpha=500, sigma=20).transpose(2,0,1)
                img, seg = red[0], red[1]

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
                img = np.concatenate((img, img, img), axis=0)
            # End Random Elastic Deformation

            d = {"image":torch.from_numpy(img).float(),
                 "mask": (torch.from_numpy(seg),
                          self.mask_to_edges(seg)),
                 "name":self.data[i]["name"]}
            return d

        elif self.split == 'val' or self.deform == False:
            img = img.unsqueeze(0)
            img = torch.cat([img, img, img], 0)
            d = {"image":img.float(),
                 "mask": (seg, self.mask_to_edges(seg)),
                 "name":self.data[i]["name"]}
            return d


    def __len__(self):
        return len(self.data)

    def mask_to_onehot(self, mask, num_classes=3):
        _mask = [mask == i for i in range(1, num_classes+1)]
        _mask = [np.expand_dims(x, 0) for x in _mask]
        return np.concatenate(_mask, 0)

    def onehot_to_binary_edges(self, mask, radius=2, num_classes=3):
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

    def mask_to_edges(self, mask):
        _edge = mask
        _edge = self.mask_to_onehot(_edge)
        _edge = self.onehot_to_binary_edges(_edge)
        return torch.from_numpy(_edge).float()

    def random_elastic_deformation(self, image, alpha, sigma, mode='nearest',
                                   random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
    ..  [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        height, width, channels = image.shape

        dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        indices = (np.repeat(np.ravel(x+dx), channels),
                np.repeat(np.ravel(y+dy), channels),
                np.tile(np.arange(channels), height*width))

        values = map_coordinates(image, indices, order=1, mode=mode)

        return values.reshape((height, width, channels))

if __name__ == '__main__':

    DATA_DIR = "/home/rexma/Desktop/MRI_Images/LCTSC" 
    augs = Compose([PaddingCenterCrop(352)])
    dataset = LungData(DATA_DIR, augmentations=augs)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for idx, batch in enumerate(dloader):
        img, mask = batch['image'], batch['mask']
        print(mask[0].shape, img.shape, mask[0].max(), mask[0].min(), img.max(), img.min())
