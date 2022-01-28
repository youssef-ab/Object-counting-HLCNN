import torch
import os
import torchvision.transforms as transforms
import localizerVgg
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plot
import visdom
import math
import argparse
from data_loader import MyDataloader
from utils import vis_MAP, detect_peaks, load_ckp, cm_jet

def parse_args():
    parser = argparse.ArgumentParser(description="Train hlcnn model")
    parser.add_argument("--root", dest='PATH', help="Dataset path", default="./dataset", type=str)
    parser.add_argument("--ckp_path", dest='CKPATH', help="path to checkpoint", default="./models/trained_model_canette_best.pt", type=str)
    parser.add_argument("--output", help="path to output density maps", default="./output", type=str)
    parser.add_argument("--class_name", help="class name", default="canette", type=str)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--detect_thr", type=float, default=0.5, help="Threshold detection")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    downsampling_ratio = 8 # Downsampling ratio
    test_dataset = MyDataloader(args.PATH, args.class_name, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = localizerVgg.localizervgg16(pretrained=True, dsr=downsampling_ratio)
    model, _, _, _, _ = load_ckp(args.CKPATH, model, None, None)
    output_dir = os.path.join(args.output, args.class_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.eval()
    model.cuda()

    vis = visdom.Visdom(server='http://localhost', port='8097')

    gi = 0
    gRi = 0
    ind = 0
    with torch.no_grad():
        for batch_idx, (im, GAM, numCar) in enumerate(test_loader):
            id_= batch_idx
            image = im.cuda()
            MAP = model(image)
            cMap = MAP[0,0,].data.cpu().numpy()
            
            cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())

            img_vis = im[0].cpu()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())

            upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im.shape[2], im.shape[3]))])
            vis_MAP(MAP, vis, 0, batch_idx, 1, upsampler)


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

            vis.image(img_vis, opts=dict(title=str(batch_idx) + '_image'))

            fark = np.sum(peakMAP) - int(numCar[0])
            gi = gi + abs(fark)
            gRi = gRi + fark*fark
            ind = ind + 1

            print(id_,'\t', np.sum(peakMAP), int(numCar[0]), '\tAE: ', abs(fark))

            upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im.shape[2] , im.shape[3]))])

            img = im.numpy()[0,]
            img = np.array(img)
            img = (img - img.min()) / (img.max() - img.min())
            plot.imshow(img.transpose((1, 2, 0)))

            M1 = MAP.data.cpu().contiguous().numpy().copy()
            M1_norm = (M1[0, ] - M1[0, ].min()) / (M1[0, ].max() - M1[0, ].min())
            a = upsampler(torch.Tensor(M1_norm))
            a = np.uint8(cm_jet(np.array(a)) * 255)
            if batch_idx > 0:
                from PIL import Image
                ima = Image.fromarray(a)
                peakMAP = np.uint8(np.array(peakMAP) * 255)
                peakI = Image.fromarray(peakMAP).convert("RGB")
                peakI = peakI.resize((1280,720))
                ima.save(output_dir + "/heatmap-" + str(batch_idx) + ".bmp")
                peakI.save(output_dir + "/peakmap-" + str(batch_idx) + ".bmp")

        print('MAE : {0:3.2f}'.format(gi / ind))
        print('RMSE : {0:3.3f}'.format(math.sqrt(gRi/ind)))



