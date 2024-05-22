# script used for brain extracting and registering a set of MRI scans to the MNI
import os
import numpy as np
from fsl.wrappers.bet import bet
from fsl.wrappers.flirt import flirt
from fsl.wrappers.fslstats import fslstats
from fsl.wrappers.misc import fslroi
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import zoom


def bet_and_reg(output_dir, input_dir, ref_file):
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


def compute_min_roi(input_dir, output_dir):
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


def compute_global_min_roi(input_dir):
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

    return min_vals, max_lens


def increase_volume(lens, factor=1):
    if factor < 1 or factor > 2:
        print("Invalid factor given; value has to be between 1 and 2")
        return lens

    new_lens = lens * factor ** (1 / 3)
    return new_lens


def extract_roi_from_brains(input_dir, output_dir, min_vals, dimensions):
    dir_list = os.listdir(input_dir)
    for file in tqdm(dir_list, total=len(dir_list)):
        if not file.endswith(".nii.gz"):
            continue

        in_file = os.path.join(input_dir, file)
        out_file = os.path.join(output_dir, file)

        x_start, y_start, z_start = min_vals
        x_size, y_size, z_size = dimensions
        fslroi(in_file, out_file, x_start, x_size, y_start, y_size, z_start, z_size)


def standardize_volume(in_dir):
    max_lens = np.array([0, 0, 0])
    files = os.listdir(in_dir)
    for file in tqdm(files, total=len(files)):
        if not file.endswith(".nii.gz"):
            continue

        in_file = os.path.join(in_dir, file)
        stats = fslstats(in_file).w.run()
        lens = stats[1:6:2]
        max_lens = np.maximum(max_lens, lens)

    return max_lens


def extract_full_volume_from_seg(seg_dir, mri_dir, out_dir, standard_lens):
    seg_list = os.listdir(seg_dir)
    mri_list = os.listdir(mri_dir)

    for seg in tqdm(seg_list, total=len(seg_list)):
        if not seg.endswith(".nii.gz"):
            continue

        seg_file = os.path.join(seg_dir, seg)
        stats = fslstats(seg_file).w.run()
        lens = stats[1:6:2]
        min_vals = stats[0:5:2]
        expand_factors = (standard_lens - lens) / 2
        new_min_vals = min_vals - expand_factors

        min_x, min_y, min_z = new_min_vals
        len_x, len_y, len_z = standard_lens

        for mri in mri_list:
            if mri != seg:
                continue

            mri_file = os.path.join(mri_dir, mri)
            out_file = os.path.join(out_dir, mri)
            fslroi(mri_file, out_file, min_x, len_x, min_y, len_y, min_z, len_z)
            break


def extract_full_volume_from_seg_2(seg_dir, mri_dir, out_dir, increase_factor=1):
    seg_list = os.listdir(seg_dir)

    for seg in tqdm(seg_list, total=len(seg_list)):
        if not seg.endswith(".nii.gz"):
            continue

        seg_file = os.path.join(seg_dir, seg)
        stats = fslstats(seg_file).w.run()
        lens = stats[1:6:2]
        min_vals = stats[0:5:2]

        if increase_factor > 1:
            new_lens = increase_volume(lens, increase_factor)
            expand_factors = (new_lens - lens) / 2
            min_vals = min_vals - expand_factors
            lens = new_lens

        min_x, min_y, min_z = min_vals
        len_x, len_y, len_z = lens

        mri = Path(seg).stem
        mri_file = os.path.join(mri_dir, mri)
        out_file = os.path.join(out_dir, seg)
        fslroi(mri_file, out_file, min_x, len_x, min_y, len_y, min_z, len_z)


def extract_roi_perfect(seg_dir, mri_dir, out_dir, increase_factor=1):
    seg_list = os.listdir(seg_dir)
    mri_list = os.listdir(mri_dir)

    for seg in tqdm(seg_list, total=len(seg_list)):
        if not seg.endswith(".nii.gz"):
            continue

        seg_file = os.path.join(seg_dir, seg)
        stats = fslstats(seg_file).w.run()
        lens = stats[1:6:2]
        min_vals = stats[0:5:2]

        if increase_factor > 1:
            new_lens = increase_volume(lens, increase_factor)
            expand_factors = (new_lens - lens) / 2
            min_vals = min_vals - expand_factors
            lens = new_lens

        min_x, min_y, min_z = min_vals
        len_x, len_y, len_z = lens

        for mri in mri_list:
            if mri != seg:
                continue

            mri_file = os.path.join(mri_dir, mri)
            out_file = os.path.join(out_dir, mri)
            fslroi(mri_file, out_file, min_x, len_x, min_y, len_y, min_z, len_z)
            break


def extract_rois(in_dir, out_dir, standard_lens):
    mri_list = os.listdir(in_dir)

    for mri in tqdm(mri_list, total=len(mri_list)):
        if not mri.endswith(".nii.gz"):
            continue

        in_file = os.path.join(in_dir, mri)
        stats = fslstats(in_file).w.run()
        lens = stats[1:6:2]
        min_vals = stats[0:5:2]
        expand_factors = (standard_lens - lens) / 2
        new_min_vals = min_vals - expand_factors

        min_x, min_y, min_z = new_min_vals
        len_x, len_y, len_z = standard_lens

        out_file = os.path.join(out_dir, mri)
        fslroi(in_file, out_file, min_x, len_x, min_y, len_y, min_z, len_z)


def resample_volume(volume, target_shape):
    current_shape = volume.shape
    scale_factors = [t / c for t, c in zip(target_shape, current_shape)]
    resampled_volume = zoom(volume, scale_factors, order=3)  # Order 3 is spline interpolation
    return resampled_volume


def resample_mris(in_dir, out_dir, target_shape):
    list_mris = os.listdir(in_dir)
    for mri in tqdm(list_mris, total=len(list_mris)):
        if not mri.endswith(".nii.gz"):
            continue

        in_mri = os.path.join(in_dir, mri)
        nii = nib.load(in_mri)
        volume = nii.get_fdata()
        resampled_volume = resample_volume(volume, target_shape)
        out_mri = os.path.join(out_dir, mri)
        resampled_nii = nib.Nifti1Image(resampled_volume, affine=nii.affine)
        nib.save(resampled_nii, out_mri)


def extract_exact_roi(in_dir, out_dir):
    files = os.listdir(in_dir)
    for file in tqdm(files, total=len(files)):
        if not file.endswith(".nii.gz"):
            continue

        in_file = os.path.join(in_dir, file)
        stats = fslstats(in_file).w.run()
        x_start, x_size, y_start, y_size, z_start, z_size, t0, t1 = stats
        out_file = os.path.join(out_dir, file)
        fslroi(in_file, out_file, x_start, x_size, y_start, y_size, z_start, z_size, t0, t1)


def normalize_volume(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    norm_volume = (volume - mean) / std
    return norm_volume


def intensity_normalization(in_dir, out_dir):
    files = os.listdir(in_dir)
    for file in tqdm(files, total=len(files)):
        if not file.endswith(".nii.gz"):
            continue

        in_file = os.path.join(in_dir, file)
        mri = nib.load(in_file)
        volume = mri.get_fdata()
        volume = normalize_volume(volume)
        out_file = os.path.join(out_dir, file)
        normalized_mri = nib.Nifti1Image(volume, affine=mri.affine)
        nib.save(normalized_mri, out_file)

def get_mean_dimensions(in_dir):
    length_sums = np.array([0, 0, 0])
    files = os.listdir(in_dir)
    for file in files:
        if ".nii" not in file:
            continue

        in_file = os.path.join(in_dir, in_file)
        stats = fslstats(in_file).w.run()
        lens = stats[1:6:2]
        length_sums += lens

    num_files = len(files)
    mean_lens = length_sums / num_files

    return (mean_lens[0], mean_lens[1], mean_lens[2])


def preprocess_mris(in_dir, out_dir, ref_file, mat_dir):
    """
    in_dir - directory where subject separated scans can be found
    out_dir - directory to contain all preprocessing directories and corresponding files
    """
    # make the brain and registrations directories
    brain_dir=f"{in_dir}/brain_extractions"
    reg_dir=f"{in_dir}/registered_brains"

    if not os.path.isdir(brain_dir):
        os.mkdir(brain_dir)

    if not os.path.isdir(reg_dir):
        os.mkdir(reg_dir)

    flirt_params = {
        'omat': mat_dir,
        'out': "",
        'dof': 12
    }

    # brain extract and register
    mri_scans = os.listdir(in_dir)
    for mri in tqdm(mri_scans, total=len(mri_scans)):
        if not mri.endswith(".nii"):
            continue

        # extract the brain
        in_mri = os.path.join(in_dir, mri)
        out_brain = os.path.join(brain_dir, mri)
        bet(in_mri, out_brain)

        # register the brain
        out_flirt = os.path.join(reg_dir, mri)
        flirt_params["out"] = out_flirt
        flirt(out_brain, ref_file, **flirt_params)

    mean_dimensions = get_mean_dimensions(reg_dir) 

    exact_roi_dir = f"{in_dir}/exact_roi"
    mean_roi_dir = f"{in_dir}/mean_roi_reg_brains"
    resampled_dir = f"{in_dir}/resampled_reg_brains"

    if not os.path.isdir(exact_roi_dir):
        os.mkdir(exact_roi_dir)

    if not os.path.isdir(mean_roi_dir):
        os.mkdir(mean_roi_dir)

    if not os.path.isdir(resampled_dir):
        os.mkdir(resampled_dir)

    # extract exact rois and resample them to mean dimensions
    extract_exact_roi(reg_dir, exact_roi_dir)    
    resample_mris(exact_roi_dir, resampled_dir, mean_dimensions)
    os.removedirs(exact_roi_dir)

    # straight away extract mean dimensions as ROIs
    extract_rois(reg_dir, mean_roi_dir, mean_dimensions)


# if __name__ == '__main__':
