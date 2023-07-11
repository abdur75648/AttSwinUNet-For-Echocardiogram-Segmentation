import os
import numpy as np
import SimpleITK as sitk
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

both_transform = A.Compose(
    [A.Resize(width=224, height=224),
     A.augmentations.geometric.transforms.Affine (translate_percent=(0.05,0.06),rotate=(-7,7), p=0.65),
    ], additional_targets={"image0": "image"},
)
transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.], std=[1.], max_pixel_value=1.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        # A.Normalize(mean=[0.], std=[1.], max_pixel_value=1.0,),
        ToTensorV2(),
    ]
)

class CAMUS_4CH_Dataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.patients_list = sorted([x for x in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir, x)) and x.startswith('patient'))])
        if split == 'train':
            self.patients_list = self.patients_list[:400]
        elif split == 'val':
            self.patients_list = self.patients_list[400:450]

    def __len__(self):
        return len(self.patients_list)

    def __getitem__(self, index):
        patient_unique_id = str(self.patients_list[index])
        patient_dir = os.path.join(self.data_dir, patient_unique_id)
        image_path = os.path.join(patient_dir, patient_unique_id + "_4CH_ED.mhd")
        mask_path = os.path.join(patient_dir, patient_unique_id + "_4CH_ED_gt.mhd")
        image_file = sitk.GetArrayFromImage(sitk.ReadImage(image_path,sitk.sitkFloat32))[0,:,:]
        mask_file = sitk.GetArrayFromImage(sitk.ReadImage(mask_path,sitk.sitkFloat32))[0,:,:]
        mask_file[mask_file != 1] = 0
        mask_file[mask_file == 1] = 1.0
        
        # dim0 = image_file.shape[0]
        # dim1 = image_file.shape[1]
        # extended_mask = np.zeros((dim0,dim1,3))
        # extended_mask[:,:,0] = np.array(mask_file==1.0,dtype=float)
        # extended_mask[:,:,1] = np.array(mask_file==2.0,dtype=float)
        # extended_mask[:,:,2] = np.array(mask_file==3.0,dtype=float)
        
        # image_file = np.repeat(image_file[..., np.newaxis], 3, -1)
    
        # No Augmentations yet.
        augmentations = both_transform(image=image_file,image0=mask_file)
        image_file,extended_mask = augmentations["image"],augmentations["image0"]
        image_file = transform_only_input(image=image_file)["image"]
        extended_mask = transform_only_mask(image=extended_mask)["image"]
        return image_file,extended_mask