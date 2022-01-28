import os
import os.path
import numpy as np
from glob import glob
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import  torchvision.transforms as transforms
import cv2
from skimage import transform as sktransform
import math
from data_aug import *
import random
import argparse


def twoD_Gaussian(m, n, amplitude, sigma_x, sigma_y):
    x = np.linspace(-m, m, 2 * m + 1)
    y = np.linspace(-n, n, 2 * n + 1)
    x, y = np.meshgrid(x, y)
    xo = 0.0
    yo = 0.0
    theta = 0.0
    offset = 0.0
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    sumg = g.sum()
    if sumg != 0:
        g /= sumg
    return g


class MyDataloader(data.Dataset):

    def __init__(self, root, class_name, train = True):
        self.root = root
        self.class_name = class_name
        self.class2idx = {"aerosol" : 0, "conserve_ronde" : 1, "conserve_rectangulaire" : 2, "canette" : 3, "sirop" : 4, "opercule" : 5}
        self.path_dataset = os.path.join(root, 'train' if train else 'valid')
        self.classes = list(self.class2idx.keys())
        self.train = train
        self.path_list_file = self.path_images(self.path_dataset)
        #print('Number of classes=%03d  number of images=%d' % (len(self.classes), len(self.path_list_file)))

    def path_images(self, path):
        class_id = self.class2idx[self.class_name]
        labels_list = glob(os.path.join(path,'*.JPG'))
        labels_class_list = []
        for img_path in labels_list :
            label_path = img_path.replace('.JPG', '.txt')
            with open(label_path, "r") as file_r :
                for line in file_r :
                    val = line.split()
                    if(int(val[0])==class_id):
                        labels_class_list.append(img_path)
                        break
        return labels_class_list
    def preprocess(self, img, min_size=1280, max_size=1280):
        H, W, C = img.shape
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        img = sktransform.resize(img, (int(H * scale), int(W * scale), C), mode='reflect', anti_aliasing=True)
        img = np.asarray(img, dtype=np.float32)
        return img

    def resize_bbox(self, bbox, out_size):
        bbox = bbox.copy()
        x_scale = float(out_size[0]) 
        y_scale = float(out_size[1]) 
        bbox[:, 0] = np.round(y_scale * bbox[:, 0])
        bbox[:, 2] = np.round(y_scale * bbox[:, 2])
        bbox[:, 1] = np.round(x_scale * bbox[:, 1])
        bbox[:, 3] = np.round(x_scale * bbox[:, 3])
        return bbox


    def read_gt_bbox(self, annoFile):
        class_id = self.class2idx[self.class_name]
        gt_boxes = []
        for line in annoFile:
            bbox = line.split()
            if int(bbox[0]) == class_id:
                gt_boxes.append([float(bbox[1])-float(bbox[3])/2, float(bbox[2])-float(bbox[4])/2, float(bbox[3]), float(bbox[4])])
        return gt_boxes

    def __getitem__(self, index):
        # id_ = '164'
        img_path = self.path_list_file[index]
        img = Image.open(img_path).convert('RGB')
        if self.train:
            if random.random() > 0.5:#0.5
                transformsColor = transforms.Compose([transforms.ColorJitter(hue=0.2, saturation=0.2)])
                img = transformsColor(img)
        img = np.asarray(img, dtype=np.float32)

        H, W, _ = img.shape

        img = self.preprocess(img)
        o_H, o_W, _ = img.shape
        dSR = 1
        GAM = np.zeros((1, int(o_H / dSR), int(o_W / dSR)))
        label_path = img_path.replace('.JPG', '.txt')
        annoFile = open(label_path, 'r')
        gt_bbox = np.asarray(self.read_gt_bbox(annoFile))

        numobj = 0
        if gt_bbox.shape[0] > 0:
            gt_boxes = np.asarray(self.resize_bbox(gt_bbox,(o_H, o_W)), dtype=np.float32)

            gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
            gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]

            if self.train:
                if random.random() > 1:
                    transforms_aug = Sequence([RandomRotate(45)])
                    img, gt_boxes = transforms_aug(img, gt_boxes)
            gt_boxes = gt_boxes / dSR
            gt_boxes[:, 0::2] = np.clip(gt_boxes[:, 0::2], 0, int(o_W / dSR))
            gt_boxes[:, 1::2] = np.clip(gt_boxes[:, 1::2], 0, int(o_H / dSR))

            gt_boxes[:, 2] = abs(gt_boxes[:, 2] - gt_boxes[:, 0])
            gt_boxes[:, 3] = abs(gt_boxes[:, 3] - gt_boxes[:, 1])


            gt_boxes = np.delete(gt_boxes, np.where(gt_boxes[:, 3]==0),0)
            gt_boxes = np.delete(gt_boxes, np.where(gt_boxes[:, 2]==0),0)

            numobj = gt_boxes.shape[0]
            # prepare GAM image

            for bbox in gt_boxes:

                bbox = np.asarray(bbox, dtype=np.int64)

                dhsizeh = int(bbox[3] / 2)
                dhsizew = int(bbox[2] / 2)

                if dhsizeh % 2 == 0:
                    dhsizeh = dhsizeh + 1

                if dhsizew % 2 == 0:
                    dhsizew = dhsizew + 1

                sigma = np.sqrt(dhsizew * dhsizeh) / (1.96*1.5)
                h_gauss = np.array(twoD_Gaussian(dhsizew, dhsizeh, sigma, math.ceil(dhsizew / 4), math.ceil(dhsizeh / 4)))
                h_gauss = h_gauss / np.max(h_gauss)

                cmin = bbox[1]
                rmin = bbox[0]
                cmax = bbox[1] + int(2*dhsizeh)+1
                rmax = bbox[0] + int(2*dhsizew)+1

                if cmax > int(o_H / dSR):
                    cmax = int(o_H / dSR)

                if rmax > int(o_W / dSR):
                    rmax = int(o_W / dSR)
                GAM[0, cmin:cmax, rmin:rmax] = GAM[0, cmin:cmax, rmin:rmax] + h_gauss[0:cmax-cmin, 0:rmax-rmin]

        downsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((int(o_H / 8), int(o_W / 8)), interpolation = transforms.InterpolationMode.LANCZOS)])
        #
        GAM = downsampler(torch.Tensor(GAM))
        GAM = np.array(GAM)
        GAM = (GAM / GAM.max()) * 1
        # plt.imshow(img)
        # # plt.show()
        # plt.imshow(GAM, cmap='gray', alpha=0.8)
        # plt.show()

        if img.ndim == 2:
            img = img[np.newaxis]
        else:
            img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        normalize = transforms.Normalize(mean=[0.39895892, 0.42411209, 0.40939609], std=[0.19080092, 0.18127358, 0.19950577])
        img = normalize(torch.from_numpy(img))

        return img, GAM, numobj

    def __len__(self):
        return len(self.path_list_file)

    def get_number_classes(self):
        return len(self.classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", dest='PATH', help="path to launch script", default="data/labels", type=str)
    args = parser.parse_args()
    dataset = MyDataloader(args.PATH, train=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    img, GAM, numobj = next(iter(train_loader))
    
    print(numobj)
    img = np.squeeze(img[0,:,:,:])
    print(img.shape)
    plt.figure(figsize=(30,10)) # specifying the overall grid size
    plt.subplot(1,2,1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(img.reshape(img.shape[1],img.shape[2],img.shape[0]))
    plt.title('image')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(GAM[0,:,:]), cmap='gray')
    plt.title('image')
    plt.axis('off')
    plt.show()
    