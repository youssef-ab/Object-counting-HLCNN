import torch
import os
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import visdom
import matplotlib as mpl
from matplotlib import pyplot as plot
import argparse
from data_loader import MyDataloader
import localizerVgg
from tqdm import tqdm
from loss import Nllloss
from utils import vis_MAP, detect_peaks, load_ckp, save_ckp

def parse_args():
    parser = argparse.ArgumentParser(description="Train hlcnn model")
    parser.add_argument("--root", dest='PATH', help="Dataset path", default="./dataset", type=str)
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument("--ckp_path", dest='CKPATH', help="path to checkpoint", default="./models/trained_model_canette.pt", type=str)
    parser.add_argument("--class_name", help="class name", default="canette", type=str)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--scheduler", type=int, default=15, help="Step size for scheduler")
    parser.add_argument("--epoch", type=int, default=35, help="Number of epochs")
    parser.add_argument("--detect_thr", type=float, default=0.5, help="Threshold detection")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    vis = visdom.Visdom(server='http://localhost', port='8097')
    model = localizerVgg.localizervgg16(pretrained=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    train_dataset = MyDataloader(args.PATH, args.class_name, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataset = MyDataloader(args.PATH, args.class_name, train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    criterionGAM = Nllloss()
    if not os.path.exists(os.path.dirname(args.ckp_path)):
        os.makedirs(os.path.dirname(args.ckp_path))
    optimizer_ft = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)
    start_epoch = 0
    min_ae = np.inf
    if args.resume :
        #ckp = os.path.join(args.CKPATH, "trained_model_" + args.class_name + '.pt')
        model, optimizer_ft, scheduler, start_epoch, min_ae = load_ckp(args.CKPATH, model, optimizer_ft, scheduler)
        if start_epoch >= args.epoch :
            print("Training is already finished, try a larger number of epochs")
            return
        print("Resume from last checkpoint.. Epoch : {0:3d}, LR : {1}, Min AE : {2:3.2f}".format(start_epoch, scheduler.get_last_lr(), min_ae))
    for epoch in range(start_epoch,args.epoch):
        train_loss = 0
        train_ae = 0
        model.train()
        for batch_idx, (data, GAM, numobj) in tqdm(enumerate(train_loader)):
            data, GAM, numobj = data.to(device, dtype=torch.float),  GAM.to(device), numobj.to(device)
            MAP = model(data)
            optimizer_ft.zero_grad()
            """if batch_idx % 20 == 0 and epoch % 1 == 0:
            img_vis = data[0].cpu()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
            vis.image(img_vis, opts=dict(title=str(epoch) + '_' + str(batch_idx) + '_image'))

            upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((data.shape[2], data.shape[3]))])
            vis_MAP(MAP, vis, epoch, batch_idx, 1, upsampler)"""

            cMap = MAP[0,0,].data.cpu().numpy()
            cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())
            cMap[cMap < args.detect_thr] = 0
            peakMAP = detect_peaks(cMap)

            MAP = MAP.view(MAP.shape[0], -1)
            GAM = GAM.view(GAM.shape[0], -1)

            fark = abs(np.sum(peakMAP) - int(numobj[0]))
            train_ae += fark*8/len(train_loader)
            loss = criterionGAM(MAP, GAM, numobj)
            train_loss += loss.item()*args.batch_size/len(train_loader)
            loss.backward()
            optimizer_ft.step()
            del data, GAM, MAP, cMap
        torch.cuda.empty_cache()
        """if batch_idx % 4 == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 'Loss: {3}\ AE:{4}'
                 .format(epoch, batch_idx, len(train_loader), loss,  abs(fark)))"""
        val_loss = 0
        val_ae = 0
        model.eval()
        for batch_idx, (data, GAM, numobj) in tqdm(enumerate(val_loader)):
            data, GAM, numobj = data.to(device, dtype=torch.float),  GAM.to(device), numobj.to(device)

            MAP = model(data)
            if batch_idx % 4 == 0 and epoch % 1 == 0:
                img_vis = data[np.random.randint(0, args.batch_size)].cpu()
                img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
                vis.image(img_vis, opts=dict(title=str(epoch) + '_' + str(batch_idx) + '_image'))

                upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((data.shape[2], data.shape[3]))])
                vis_MAP(MAP, vis, epoch, batch_idx, 1, upsampler)

            cMap = MAP[0,0,].data.cpu().numpy()
            cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())
            cMap[cMap < args.detect_thr] = 0
            peakMAP = detect_peaks(cMap)

            MAP = MAP.view(MAP.shape[0], -1)
            GAM = GAM.view(GAM.shape[0], -1)

            fark = abs(np.sum(peakMAP) - int(numobj[0]))
            val_ae += fark*2/len(val_loader)
            loss = criterionGAM(MAP, GAM, numobj)
            val_loss += loss.item()*args.batch_size/len(val_loader)
            del data, GAM, MAP, cMap
            torch.cuda.empty_cache()
        
        """print('Epoch: [{0}][{1}/{2}]\t' 'Val Loss: {3}\ Val AE:{4}'
            .format(epoch, batch_idx, len(val_loader), loss,  abs(fark)))"""
        
        scheduler.step()
        print('Epoch : {0:3d}\t' 'LR : {1}\t' 'train loss : {2:.3f}\ train AE : {3:3.2f}\t' 'Val loss : {4:.3f}\ Val AE : {5:3.2f}'
                .format(epoch, scheduler.get_last_lr(), train_loss,  abs(train_ae), val_loss,  abs(val_ae)))
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer_ft.state_dict(),
            'scheduler': scheduler.state_dict(),
            'min_ae' : min_ae
        }
        save_ckp(checkpoint, 0, args.CKPATH)
        if val_ae < min_ae :
            min_ae = val_ae
            checkpoint["min_ae"] = val_ae
            print("Epoch : {0:3d}, LR : {1}, Loss : {2:.3f}, AE : {3:3.2f}, Saving best model...".format(epoch, scheduler.get_last_lr(), val_loss, val_ae))
            save_ckp(checkpoint, 1, args.CKPATH)

if __name__ == '__main__':
    main()