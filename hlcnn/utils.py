import torch
import os
import numpy as np
import shutil
import matplotlib as mpl
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

cm_jet = mpl.cm.get_cmap('jet')

def save_ckp(state, is_best, checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        best_fpath, ext = os.path.splitext(checkpoint_path)
        best_fpath = best_fpath + "_best" + ext
        shutil.copyfile(checkpoint_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None : 
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None : 
        scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['min_ae']

def detect_peaks(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks

def vis_MAP(MAP, vis, epoch, batch_idx, mapId, upsampler):
    M1 = MAP.data.cpu().contiguous().numpy().copy()
    M1_norm = (M1[0,] - M1[0,].min()) / (M1[0,].max() - M1[0,].min())
    b = upsampler(torch.Tensor(M1_norm))
    b = np.uint8(cm_jet(np.array(b)) * 255)
    vis.image(np.transpose(b, (2, 0, 1)), opts=dict(
        title=str(epoch) + '_' + str(batch_idx) + '_' + str(mapId) + '_heatmap'))
