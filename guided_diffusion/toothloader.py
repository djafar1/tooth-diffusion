import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
import pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from skimage.morphology import ball

class ToothVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, metadata_path, test_flag=False, normalize=None, mode='train', img_size=256, augment_missing_teeth=False, reconstruct_3_mode=False):
        super().__init__()
        self.mode = mode
        self.augment_missing_teeth = augment_missing_teeth        
        self.reconstruct_3_mode = reconstruct_3_mode

        
        self.directory = os.path.expanduser(directory)
        self.metadata_path = os.path.expanduser(metadata_path)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.database = []
        
        self.image_dir = os.path.join(directory, "Images")
        self.label_dir = os.path.join(directory, "Labels")
        
        #Check for meta data
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"metadata.csv no found at {self.metadata_path}")
        self.metadata = pd.read_excel(self.metadata_path)
        self.metadata.columns = self.metadata.columns.str.strip()  # ← removes leading/trailing spaces
     
        for f in os.listdir(self.image_dir):
            if not f.endswith(".nii.gz"):
                continue
            label_path = os.path.join(self.label_dir, f)
            image_path = os.path.join(self.image_dir, f)
            if os.path.exists(label_path):
                self.database.append({
                    "image": image_path,
                    "label": label_path,
                    "name": f
                })

        print(f"Found {len(self.database)} samples in '{mode}' set.")
        
    #Normalize
    def normalize_image(self, x):
        return ((x - x.min()) / (x.max() - x.min()))
    
    
    def __getitem__(self, x):
        filedict = self.database[x]
        
        image_np = (nibabel.as_closest_canonical(nibabel.load(filedict["image"]))).get_fdata()
        label_np = (nibabel.as_closest_canonical(nibabel.load(filedict["label"]))).get_fdata()

        image_np = self.normalize_image(image_np)
        label_np = label_np.astype(np.uint8)
        
        label_np[label_np > 32] = 0 # Ensure labels are in the range 0-32

        # Name and conditions
        filename = os.path.basename(filedict['name']) 
        basename = filename  
        row_metadata = self.metadata[self.metadata['PatientID_Date.nii.gz'] == basename]
        if row_metadata.empty:
            raise KeyError(f"{basename} not found in metadata")
        
        row_metadata = row_metadata.iloc[0] # Convert to Series
        
        # Vector 1: Tooth presence 1-32 teeth
        tooth_presence = torch.zeros(32, dtype=torch.float32)
        for i in range(1, 33):
            if (label_np == i).any():
                tooth_presence[i - 1] = 1.0

        # Vectors 2–5: Metadata-based binary conditions 
        def get_vector(col_name):
            cell_value = row_metadata[col_name]
            vector = torch.zeros(32, dtype=torch.float32)
            if pd.isna(cell_value):
                return vector
            for entry in str(cell_value).split(','):
                entry = entry.strip()
                if entry.isdigit():
                    idx = int(entry)
                    if 1 <= idx <= 32:
                        vector[idx - 1] = 1.0
            return vector
        
        # Extracting vectors from metadata
        vectors = {
            "tooth_presence": tooth_presence,
            #"crown_fill": get_vector('1.crown filling'),
            #"root_crown": get_vector('2.root and crown filling'),
            #"bridge": get_vector('3.bridge'),
            #"implant": get_vector('4.implant'),
        }

        # Make copy to ensure that it doesn't modify each other
        target_image_np = image_np.copy()
        target_label_np = label_np.copy()
        cond_image_np = image_np.copy()
        cond_label_np = label_np.copy()

        present_indices = [i for i in range(32) if tooth_presence[i] == 1]

        if self.mode == 'train':
            if self.reconstruct_3_mode:
                scenario = np.random.choice(["regular", "remove", "add"], p=[1/3, 1/3, 1/3])
                if scenario == "remove" and present_indices:
                    max_remove = max(1, int(np.ceil(0.5 * len(present_indices))))
                    num = np.random.randint(1, max_remove + 1)
                    removed_teeth = np.random.choice(present_indices, num, replace=False)
                    
                    # Remove the teeth from the target image, KEEP original label, so we can we penalize for not removing them
                    # Remove tooth from presence vector, so we can penalize for not removing them
                    
                    target_image_np = inpaint_teeth(target_image_np, target_label_np, [idx+1 for idx in removed_teeth])
                    for idx in removed_teeth:
                        vectors["tooth_presence"][idx] = 0.0
                        
                        
                elif scenario == "add" and present_indices:
                    max_remove = max(1, int(np.ceil(0.5 * len(present_indices))))
                    num = np.random.randint(1, max_remove + 1)
                    removed_teeth = np.random.choice(present_indices, num, replace=False)
                    
                    # Remove the teeth from the cond image, KEEP original label, so we can we penalize for not removing them
                    # Keep original vector precence, so we can penalize for not adding them
                    
                    cond_image_np = inpaint_teeth(cond_image_np, cond_label_np, [idx+1 for idx in removed_teeth])
                    for idx in removed_teeth:
                        cond_label_np[cond_label_np == (idx+1)] = 0
                        
                # regular: do nothing
                else:
                    pass
            elif self.augment_missing_teeth:
                if np.random.rand() < 0.5 and present_indices:
                    max_remove = max(1, int(np.ceil(0.50 * len(present_indices))))
                    num = np.random.randint(1, max_remove + 1)
                    removed_teeth = np.random.choice(present_indices, num, replace=False)
                    target_image_np = inpaint_teeth(target_image_np, target_label_np, [idx+1 for idx in removed_teeth])
                    for idx in removed_teeth:
                        target_label_np[target_label_np == (idx+1)] = 0
                        vectors["tooth_presence"][idx] = 0.0

        
        if not self.mode == 'fake':
            image_tensor = torch.Tensor(target_image_np)
            label_tensor = torch.Tensor(target_label_np)

            cond_image_tensor = torch.Tensor(cond_image_np)
            cond_label_tensor = torch.Tensor(cond_label_np)

            image = torch.zeros(1, 256, 256, 256)
            label = torch.zeros(1, 256, 256, 256)
            cond_image = torch.zeros(1, 256, 256, 256)
            cond_label = torch.zeros(1, 256, 256, 256)

            image[:, :, :, :] = image_tensor
            label[:, :, :, :] = label_tensor
            cond_image[:, :, :, :] = cond_image_tensor
            cond_label[:, :, :, :] = cond_label_tensor

            if self.img_size == 128:
                downsample = nn.AvgPool3d(kernel_size=2, stride=2)
                image = downsample(image)
                label = downsample(label).long()
                cond_image = downsample(cond_image)
                cond_label = downsample(cond_label).long()
            else:
                label = label.long()
                cond_label = cond_label.long()
        else:
            image = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(label_np, dtype=torch.long).unsqueeze(0)
            cond_image = torch.tensor(cond_image_np, dtype=torch.float32).unsqueeze(0)
            cond_label = torch.tensor(cond_label_np, dtype=torch.long).unsqueeze(0)

        image = self.normalize(image)
        cond_image = self.normalize(cond_image)     
        
        # Adding the random flipping of image and label       
        if self.mode == 'train' and np.random.rand() < 0.5:
            # flip horizontally // mirror
            image = torch.flip(image, dims=[1])
            cond_image = torch.flip(cond_image, dims=[1])
            label = torch.flip(label, dims=[1])
            cond_label = torch.flip(cond_label, dims=[1])

            upper = (label > 0) & (label <= 16)
            lower = (label >= 17) & (label <= 32)

            label[upper] = 17 - label[upper]
            label[lower] = 49 - label[lower]

            cond_label[upper] = 17 - cond_label[upper]
            cond_label[lower] = 49 - cond_label[lower]

                        
        if self.mode in ['fake', 'eval']:
            return {
                "image": image,
                "label": label,
                "cond_image": cond_image,
                "cond_label": cond_label,
                "name": [basename],
                "tooth_presence": vectors["tooth_presence"],
                #"crown_fill": vectors["crown_fill"],
                #"root_crown": vectors["root_crown"],
                #"bridge": vectors["bridge"],
                #"implant": vectors["implant"],
            }

        return {
            "image": image,
            "label": label,
            "cond_image": cond_image,
            "cond_label": cond_label,
            "tooth_presence": vectors["tooth_presence"],
            #"crown_fill": vectors["crown_fill"],
            #"root_crown": vectors["root_crown"],
            #"bridge": vectors["bridge"],
            #"implant": vectors["implant"],
        }
    def __len__(self):
        return len(self.database)

def inpaint_teeth(image_np, label_np, tooth_ids, sphere_radius=2):
    struct = ball(sphere_radius)

    tooth_mask = np.zeros_like(label_np, dtype=bool)
    for tooth_id in tooth_ids:
        tooth_mask |= (label_np == tooth_id)

    tooth_mask = binary_dilation(tooth_mask, structure=struct)
    teeth_mask = binary_dilation(label_np > 0, structure=struct)

    V1 = image_np.copy()
    V1[teeth_mask] = np.nan

    missing = np.isnan(V1)
    dist, (inds_z, inds_y, inds_x) = distance_transform_edt(missing, return_indices=True)

    V2 = image_np.copy()
    V2[teeth_mask] = image_np[inds_z[teeth_mask],inds_y[teeth_mask],inds_x[teeth_mask]]

    V2_smooth = gaussian_filter(V2, sigma=1.0)

    inpainted = image_np.copy()
    inpainted[tooth_mask] = V2_smooth[tooth_mask]

    return inpainted