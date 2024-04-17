# script used for brain extracting and registering a set of MRI scans to the MNI
import os
import numpy as np
from fsl.wrappers.bet import bet
from fsl.wrappers.flirt import flirt
from fsl.wrappers.fslstats import fslstats
from fsl.wrappers.misc import fslroi
from tqdm import tqdm


def bet_and_reg():
    output_dir = "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/RegShortDataset"
    input_dir = "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/ShortDataset"
    ref_file = "/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
    dir_list = os.listdir(input_dir)

    for i in tqdm(range(len(dir_list)), total=len(dir_list)):
        file = dir_list[i]
        if not file.endswith(".nii"):
            continue

        in_file = os.path.join(input_dir, file)
        out_bet = os.path.join(output_dir, file + "_bet")
        out_flirt = os.path.join(output_dir, file)

        bet(in_file, out_bet)

        flirt_params = {
            'omat': "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/GarbageMatricies/out.mat",
            'out': out_flirt,
            'dof': 12
        }
        flirt(out_bet, ref_file, **flirt_params)

    print("Preprocessing is done!")


def compute_min_roi():
    input_dir = "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/RegShortDataset"
    output_dir = "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/RegMinROIDataset"
    dir_list = os.listdir(input_dir)

    min_vals = [999, 999, 999]
    max_vals = [0, 0, 0]

    # computing minimum and maximum coordinates
    print("Computing min and max values...")
    for file in tqdm(dir_list, total=len(dir_list)):
        if not file.endswith(".nii.gz"):
            continue

        in_file = os.path.join(input_dir, file)
        stats = fslstats(in_file).w.run()
        vals = stats[0:5:2]
        lens = stats[1:6:2]
        vals_up = vals + lens

        min_vals = np.minimum(min_vals, vals)
        max_vals = np.maximum(max_vals, vals_up)

    max_lens = max_vals - min_vals

    # extracting the regions of interest
    print("Extracting regions of interest...")
    for file in tqdm(dir_list, total=len(dir_list)):
        if not file.endswith(".nii.gz"):
            continue

        in_file = os.path.join(input_dir, file)
        out_file = os.path.join(output_dir, file)

        x_start, y_start, z_start = min_vals
        x_size, y_size, z_size = max_lens
        fslroi(in_file, out_file, x_start, x_size, y_start, y_size, z_start, z_size)


compute_min_roi()
