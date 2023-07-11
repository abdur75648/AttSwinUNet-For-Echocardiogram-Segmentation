import os,torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure, morphology
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from attention_swin_unet.attention_swin_unet import SwinAttentionUnet as SwinAttnUNetModel
from camus import CAMUS_4CH_Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../scratch/data/Camus/training"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 8
CHANNELS_IMG = 1  # 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
TRAINED_PATH = "../scratch/attn_swin_unet/Epoch_50_gen.pth.tar"
IMAGE_SIZE = 224

def dice_coeff(pred,target,smooth=100):
    target = (target>0.5).float()
    pred = (pred>0.5).float()
    pred_f = pred.flatten()
    target_f = target.flatten()
    intersection = torch.sum(pred_f*target_f)
    return (2. * intersection +smooth)/(torch.sum(pred_f*pred_f)+torch.sum(target_f*target_f)+smooth)

def val_eval(gen,loader):
    loop = tqdm(loader,leave=True)
    LV_dice_accum=0
    steps=0
    for idx,(x,y) in enumerate(loop):
        x,y = x.to(DEVICE),y.to(DEVICE)
        with torch.no_grad():
            y_fake = gen(x)
        LV_add=0
        times = int(y.size()[0])
        for i in range(times):
            LV_add+=float(dice_coeff(y_fake[i,0,:,:],y[i,0,:,:]))
        LV_dice_accum+= LV_add/times
        steps+=1
    print("Avg LV dice score for validation is: ",LV_dice_accum/steps)

print("Device: ", DEVICE)
print("Loading Data...")
# train_data = CAMUS_4CH_Dataset(DATA_DIR,split='train')
validation_data = CAMUS_4CH_Dataset(DATA_DIR,split='val')
print("Validation Dataset Size: ", len(validation_data))
print("Creating Data Loader")
val_loader = DataLoader(validation_data,batch_size=1,shuffle=False)
print("Dataloaders with train batch_size ",BATCH_SIZE," and val batch_size 1 created!")

print("Creating Model")
model =  SwinAttnUNetModel(in_chans=1).to(DEVICE)
if TRAINED_PATH is not None:
    model.load_state_dict(torch.load(TRAINED_PATH))
    print("Loaded weights from ", TRAINED_PATH)

val_eval(model,val_loader)