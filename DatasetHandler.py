from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pandas as pd
import os


class MriDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        mri_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # convert from Nifti1 to numpy - this can be done as part of the transform
        mri = nib.load(mri_path)
        mri = np.array(mri.dataobj, dtype=np.ubyte)
        # remove the 4th dimension
        if mri.ndim == 4:
            mri = mri[:, :, :, 0]
        # mri = mri[np.newaxis, ...] # adds the channel dimension
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            mri = self.transform(mri)
        if self.target_transform:
            label = self.target_transform(label)
        return mri, label

    def get_labels(self, indices=None):
        if indices is None:
            return self.img_labels.iloc[:, 1].to_numpy()

        return self.img_labels.iloc[indices, 1].to_numpy()

