import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import localizerVgg
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plot
import visdom
import argparse
from utils import vis_MAP, detect_peaks, load_ckp, cm_jet
from skimage import transform as sktransform
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Train hlcnn model")
    parser.add_argument("--img_path", help="Image path", default="./dataset/test/sac1/DSC09110.JPG", type=str)
    parser.add_argument("--ckp_path", help="path to checkpoint", default="./models/trained_model_canette_best.pt", type=str)
    parser.add_argument("--output", help="path to output density maps", default="./output", type=str)
    parser.add_argument("--class_name", help="class name", default="canette", type=str)
    parser.add_argument("--detect_thr", type=float, default=0.39, help="Threshold detection")
    args = parser.parse_args()
    return args

def preprocess(img, min_size=720, max_size=1280):
    H, W, C = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktransform.resize(img, (int(H * scale), int(W * scale), C), mode='reflect', anti_aliasing=True)
    img = np.asarray(img, dtype=np.float32)
    return img

if __name__ == '__main__':
    args = parse_args()
    downsampling_ratio = 8 # Downsampling ratio
    # Load image 
    img = Image.open(args.img_path).convert('RGB')
    img = np.asarray(img, dtype=np.float32)
    img = preprocess(img)
    img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img = img[np.newaxis]
    normalize = transforms.Normalize(mean=[0.39895892, 0.42411209, 0.40939609], std=[0.19080092, 0.18127358, 0.19950577])
    img = normalize(torch.from_numpy(img))
    model = localizerVgg.localizervgg16(pretrained=True, dsr=downsampling_ratio)
    model, _, _, _, _ = load_ckp(args.ckp_path, model, None, None)
    output_dir = os.path.join(args.output, args.class_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name, ext = os.path.splitext(os.path.basename(args.img_path))

    model.eval()
    model.cuda()

    vis = visdom.Visdom(server='http://localhost', port='8097')

    with torch.no_grad():
        image = img.cuda()
        MAP = model(image)
        cMap = MAP[0,0,].data.cpu().numpy()
            
        cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())
        
        img_vis = img[0].cpu()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())

        upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img.shape[2], img.shape[3]))])
        vis_MAP(MAP, vis, 0, 0, 1, upsampler)
        cMap[cMap < args.detect_thr] = 0
        peakMAP = detect_peaks(cMap)
        arrX = np.where(peakMAP)[0]
        arrY = np.where(peakMAP)[1]
        for i in range(0, arrX.shape[0]):
            for k in range(-2, 2):
                for j in range(-2, 2):
                    img_vis[0, arrX[i]*downsampling_ratio+k, arrY[i]*downsampling_ratio+j] = 1
                    img_vis[1, arrX[i]*downsampling_ratio+k, arrY[i]*downsampling_ratio+j] = 0
                    img_vis[2, arrX[i]*downsampling_ratio+k, arrY[i]*downsampling_ratio+j] = 0

        vis.image(img_vis, opts=dict(title=str(0) + '_image'))
        print("Nombre d'objets détectés",'\t', np.sum(peakMAP))
        upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img.shape[2] , img.shape[3]))])

        im= img.numpy()[0,]
        im = np.array(im)
        im = (im - im.min()) / (im.max() - im.min())
        plot.imshow(im.transpose((1, 2, 0)))

        M1 = MAP.data.cpu().contiguous().numpy().copy()
        M1_norm = (M1[0, ] - M1[0, ].min()) / (M1[0, ].max() - M1[0, ].min())
        a = upsampler(torch.Tensor(M1_norm))
        a = np.uint8(cm_jet(np.array(a)) * 255)
        ima = Image.fromarray(a)
        peakMAP = np.uint8(np.array(peakMAP) * 255)
        peakI = Image.fromarray(peakMAP).convert("RGB")
        peakI = peakI.resize((1280,720))
        ima.save(output_dir + "/heatmap-" + image_name + ".bmp")
        peakI.save(output_dir + "/peakmap-" + image_name + ".bmp")