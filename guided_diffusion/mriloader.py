import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class MRIVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, metadata_path, test_flag=False, normalize=None, mode='train', img_size=256):
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.metadata_path = os.path.expanduser(metadata_path)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.database = []
        
        # We have subfolders containing images and brain
        self.skull_dir = os.path.join(self.directory, 'skull')
        self.brain_dir = os.path.join(self.directory, 'brain')
        
        #Check for meta data
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"metadata.csv no found at {self.metadata_path}")
        self.metadata = pd.read_csv(self.metadata_path)
        
        #Histogram hardcoded 5.
        self.histogram_bin = KBinsDiscretizer(n_bins=5, encode='onehot-dense')
        ages = self.metadata['Age'].values.reshape(-1 , 1)
        self.histogram_bin.fit(ages)

        skull_files = []
        for f in os.listdir(self.skull_dir):
            skull_files.append(f)
        
        for image in skull_files:
            base_name = image.replace("skull_", "", 1) # get base name
            brain_file = "brain_" + base_name # get corresponding brain file
            brain_path = os.path.join(self.brain_dir, brain_file)
            if os.path.exists(brain_path):
                datapoint = dict()
                datapoint['skull'] = os.path.join(self.skull_dir, image)
                datapoint['brain'] = brain_path
                datapoint['base_name'] = base_name
                self.database.append(datapoint)

        print("Found {} items in dataset".format(len(self.database)))
        
    #Normalize
    def normalize_image(self, x):
        return ((x - x.min()) / (x.max() - x.min()))
    
    def __getitem__(self, x):
        filedict = self.database[x]
        
        #image
        skull_name = filedict['skull']
        skull_nib_img = nibabel.as_closest_canonical(nibabel.load(skull_name))
        skull_out = skull_nib_img.get_fdata()
        #Normalize to range 0 to 1
        skull_out = self.normalize_image(skull_out)
        
        #brain
        brain_name = filedict['brain']
        brain_nib_img = nibabel.as_closest_canonical(nibabel.load(brain_name))
        brain_out = brain_nib_img.get_fdata()
        #Normalize to range 0 to 1
        brain_out = self.normalize_image(brain_out)
        
        if not self.mode == 'fake':
            skull_out = torch.Tensor(skull_out)
            brain_out = torch.Tensor(brain_out)

            skull_image = torch.zeros(1, 256, 256, 256)
            skull_image[:, :, :, :] = skull_out
            
            brain_image = torch.zeros(1, 256, 256, 256)
            brain_image[:, :, :, :] = brain_out

            if self.img_size == 128:
                downsample = nn.AvgPool3d(kernel_size=2, stride=2)
                skull_image = downsample(skull_image)
                brain_image = downsample(brain_image)
        else:
            skull_image = torch.tensor(skull_out, dtype=torch.float32)
            skull_image = skull_image.unsqueeze(dim=0)
            brain_image = torch.tensor(brain_out, dtype=torch.float32)
            brain_image = brain_image.unsqueeze(dim=0)

        # normalization function from generation train.py appliad usually (* 2 - 1)
        skull_image = self.normalize(skull_image)
        brain_image = self.normalize(brain_image)
        
        filename = os.path.basename(filedict['base_name']) #To get the name only not full path
        basename = filename.split('_0000.nii.gz')[0]
        row_metadata = self.metadata[self.metadata['BASENAME'] == basename]
        if row_metadata.empty:
            raise KeyError(f"BASENAME {basename} not found in the metadata")

        metadata = row_metadata.iloc[0]
        sex = [0, 1] if metadata['Sex'] == 'F' else [1, 0]
        
        #Using the histogram binning [0], so pass it as [] not [[]]
        age = self.histogram_bin.transform(np.array([[metadata['Age']]]))[0]
        
        if metadata['Screen.Diagnosis'] == 'AD':
            disease = [1, 0, 0]
        elif metadata['Screen.Diagnosis'] == 'CN':
            disease = [0, 1, 0]
        elif metadata['Screen.Diagnosis'] == 'MCI':
            disease = [0, 0, 1]
        else:
            raise KeyError(f"Invalid screen diagnosis value {metadata['Screen.Diagnosis']}")
        
        
        if self.mode == 'fake':
            return{
                'skull': skull_image,
                'brain': brain_image,
                'name': basename,
                'sex': torch.tensor(sex, dtype=torch.float32),
                'age': torch.tensor(age, dtype=torch.float32),
                'disease': torch.tensor(disease, dtype=torch.float32)

            }
        elif self.mode == "eval":
            return{
                'skull': skull_image,
                'brain': brain_image,
                'name': basename,
                'sex': torch.tensor(sex, dtype=torch.float32),
                'age': torch.tensor(age, dtype=torch.float32),
                'disease': torch.tensor(disease, dtype=torch.float32)

            }
        else:
            return{
                'skull': skull_image,
                'brain': brain_image,
                'sex': torch.tensor(sex, dtype=torch.float32),
                'age': torch.tensor(age, dtype=torch.float32),
                'disease': torch.tensor(disease, dtype=torch.float32)
            }
    def __len__(self):
        return len(self.database)