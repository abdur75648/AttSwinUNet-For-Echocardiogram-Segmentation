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
SAVE_DIR = "../scratch/attn_swin_unet"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 8
CHANNELS_IMG = 1  # 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
TRAINED_PATH = None # "../scratch/attn_swin_unet/Epoch_500_gen.pth.tar"
IMAGE_SIZE = 224

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

def dice_coeff(pred,target,smooth=100):
    target = (target>0.5).float()
    pred = (pred>0.5).float()
    pred_f = pred.flatten()
    target_f = target.flatten()
    intersection = torch.sum(pred_f*target_f)
    return (2. * intersection +smooth)/(torch.sum(pred_f*pred_f)+torch.sum(target_f*target_f)+smooth)

def save_some_examples(model, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred = (pred>0.5).float()
        save_image(pred, folder + f"/y_gen_{epoch}.png")
        save_image(x , folder + f"/input_{epoch}.png")
        save_image(y , folder + f"/gt_{epoch}.png")
    model.train()
    
def save_checkpoint(model, filename="Epoch_1_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(model.state_dict(), filename)

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
train_data = CAMUS_4CH_Dataset(DATA_DIR,split='train')
validation_data = CAMUS_4CH_Dataset(DATA_DIR,split='val')
print("Train Dataset Size: ", len(train_data))
print("Validation Dataset Size: ", len(validation_data))
print("Creating Data Loaders")
train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
val_loader = DataLoader(validation_data,batch_size=1,shuffle=False)
print("Dataloaders with train batch_size ",BATCH_SIZE," and val batch_size 1 created!")

print("Creating Model")
model =  SwinAttnUNetModel(in_chans=1).to(DEVICE)
if TRAINED_PATH is not None:
    model.load_state_dict(torch.load(TRAINED_PATH))
    print("Loaded weights from ", TRAINED_PATH)

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
criteria  = torch.nn.BCEWithLogitsLoss()

print("Starting Training...")
model.train()
L1_loss_per_epoch=[]
Dice_coeff_LV_per_epoch=[]
steps=0
grad_scaler = torch.cuda.amp.GradScaler()
for epoch in tqdm(range(NUM_EPOCHS)):
    print("Epoch number",epoch+1,"!")
    
    train_loop = tqdm(train_loader,leave=True)
    L1_accum=0
    LV_dice_accum=0
    steps=0
    for idx,(x,y) in enumerate(train_loop):
        steps+=1
        x,y = x.to(DEVICE),y.to(DEVICE)
        
        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = criteria(pred, y)
        optimizer.zero_grad()
        
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        
        
        # loss.backward()
        # optimizer.step()
        L1_accum += loss.item()
        
        # Calculating Dice Score
        LV_add=0
        times = int(y.size()[0])
        for i in range(times):
            LV_add+=float(dice_coeff(pred[i,0,:,:],y[i,0,:,:]))
        LV_dice_accum+= LV_add/times
        
    L1_loss_per_epoch.append(L1_accum/steps)
    Dice_coeff_LV_per_epoch.append(LV_dice_accum/steps)
    print("Avg L1 loss this epoch is: ",L1_accum/steps)
    print("Avg LV dice score for this epoch is: ",LV_dice_accum/steps)
    
    
    if (epoch+1)%50==0:
        val_eval(model,val_loader)
        save_some_examples(model, val_loader, epoch, folder=SAVE_DIR)
        save_checkpoint(model, filename= os.path.join(SAVE_DIR,"Epoch_" + str(epoch+1) + ".pth.tar"))

print("Training Completed")
plt.title("L1 loss")
plt.plot(L1_loss_per_epoch)
plt.savefig(os.path.join(SAVE_DIR,"L1 loss.png"))

val_eval(model,val_loader)