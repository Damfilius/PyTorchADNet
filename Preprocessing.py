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
import shutil
import subprocess
from Utils import create_dir


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
    sum_lens = np.array([0.0, 0.0, 0.0])
    files = os.listdir(in_dir)

    for file in tqdm(files, total=len(files)):
        if not file.endswith(".nii.gz"):
            continue

        in_file = os.path.join(in_dir, file)
        stats = fslstats(in_file).w.run()
        x_start, x_size, y_start, y_size, z_start, z_size, t0, t1 = stats
        sum_lens += [x_size, y_size, z_size]
        out_file = os.path.join(out_dir, file)
        fslroi(in_file, out_file, x_start, x_size, y_start, y_size, z_start, z_size)

    num_files = len(files)
    mean_dimensions = np.ceil(sum_lens / num_files)
    return (mean_dimensions[0], mean_dimensions[1], mean_dimensions[2])


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
    length_sums = np.array([0.0, 0.0, 0.0])
    files = os.listdir(in_dir)
    for file in tqdm(files, total=len(files)):
        if ".nii" not in file:
            continue

        in_file = os.path.join(in_dir, file)
        stats = fslstats(in_file).w.run()
        lens = stats[1:6:2]
        length_sums += lens

    num_files = len(files)
    mean_lens = length_sums / num_files

    return mean_lens


# wrapper for the run_first_all function since the original wrapper doesn't work
def run_first_all_bet(input, output, report, is_brain=False):
    command = [
        'timeout',
        '-s',
        'SIGKILL',
        '10m',
        'run_first_all',
        '-i', input,
        '-o', output
    ]

    if is_brain:
        command.append('-b')

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Segmentation on {input} performed successfully...", file=report)
        print("Output:", result.stdout.decode(), file=report)
        print("Errors:", result.stderr.decode(), file=report)
    except subprocess.CalledProcessError as e:
        print("Error executing command:")
        print("Output:", e.stdout.decode(), file=report)
        print("Errors:", e.stderr.decode(), file=report)


def perform_segmentations(in_dir, out_dir, report, is_brain=False):
    scans = os.listdir(in_dir)
    for scan in tqdm(scans, total=len(scans)):
        if ".nii" not in scan:
            continue

        # prepare the arguments
        # perform the segmentation
        in_scan = os.path.join(in_dir, scan)
        scan_no_ext = Path(Path(scan).stem).stem
        out_seg = os.path.join(out_dir, scan_no_ext)
        run_first_all_bet(in_scan, out_seg, report, is_brain)

    # remove redundant files
    seg_items = os.listdir(out_dir)
    for seg_item in seg_items:
        if "firstseg" in seg_item:
            continue

        item_to_remove = os.path.join(out_dir, seg_item)
        if os.path.isdir(item_to_remove):
            shutil.rmtree(item_to_remove)
        else:
            os.remove(item_to_remove)


def get_mri_from_seg(seg_file):
    parts = seg_file.split('_')
    mri_name = f"{parts[0]}_{parts[1]}.nii.gz"
    return mri_name


def get_roi(file, growth_factor):
    stats = fslstats(file).w.run()
    lens = stats[1:6:2]
    min_vals = stats[0:5:2]

    if growth_factor > 1:
        new_lens = increase_volume(lens, growth_factor)
        expand_factors = (new_lens - lens) / 2
        min_vals = min_vals - expand_factors
        lens = new_lens

    return min_vals, lens


def get_roi_volumes_from_segs(seg_dir, mri_dir, out_dir, growth_factor=1):
    seg_files = os.listdir(seg_dir)
    mri_files = os.listdir(mri_dir)
    sum_lens = np.array([0.0, 0.0, 0.0])

    for seg_file in tqdm(seg_files, total=len(seg_files)):
        if not seg_file.endswith(".nii.gz"):
            continue

        full_seg_file = os.path.join(seg_dir, seg_file)
        min_vals, lens = get_roi(full_seg_file, growth_factor)
        sum_lens += lens

        found = False
        mri_file = get_mri_from_seg(seg_file)
        for mri in mri_files:
            if mri == mri_file:
                found = True
                break

        if not found:
            print(f"Could not find file {mri_file} in directory {mri_dir}")
            continue

        # extract roi from mri file
        full_mri_file = os.path.join(mri_dir, mri_file)
        out_volume = os.path.join(out_dir, mri_file)
        fslroi(full_mri_file, out_volume, min_vals[0], lens[0], min_vals[1], lens[1], min_vals[2], lens[2])

    mean_dimensions = np.ceil(sum_lens / len(seg_files))
    return (mean_dimensions[0], mean_dimensions[1], mean_dimensions[2])


def bet_and_reg(in_dir, bet_out_dir, reg_out_dir, mat_dir, ref_file):
    flirt_params = {
        'omat': f"{mat_dir}/outmat.mat",
        'out': "",
        'dof': 12
    }
    mri_scans = os.listdir(in_dir)

    for mri in tqdm(mri_scans, total=len(mri_scans)):
        if not mri.endswith(".nii"):
            continue

        # extract the brain
        in_mri = os.path.join(in_dir, mri)
        out_brain = os.path.join(bet_out_dir, mri)
        bet(in_mri, out_brain)

        # register the brain
        out_flirt = os.path.join(reg_out_dir, mri)
        flirt_params["out"] = out_flirt
        out_brain = f"{out_brain}.gz"
        flirt(out_brain, ref_file, **flirt_params)


def remove_seg_waste(brain_dir):
    files = os.listdir(brain_dir)
    num_brains = 0

    for file in files:
        if "std_sub" not in file:
            num_brains += 1
            continue

        file_to_rm = os.path.join(brain_dir, file)
        os.remove(file_to_rm)

    return num_brains


# addition for resampled_reg_brains, etc.
def calculate_inter_fold_dimensions(folds_dir, addition):
    global_mean_dimensions = np.array([0.0, 0.0, 0.0])
    folds = os.listdir(folds_dir)

    for fold in folds:
        fold_dir = os.path.join(folds_dir, fold)
        resampled_dir = os.path.join(fold_dir, addition)
        global_mean_dimensions += get_mean_dimensions(resampled_dir)

    global_mean_dimensions /= len(folds)
    return global_mean_dimensions



def resampled_all_resampled_brains(folds_dir):
    volumes = np.array([])
    folds = os.listdir(folds_dir)

    for fold in folds:
        fold_dir = os.path.join(folds_dir, fold)
        resampled_dir = os.path.join(fold_dir, "resampled_reg_brains")
        fold_files = os.listdir(resampled_dir)
        random_file = fold_files[0]
        fold_file_path = os.path.join(resampled_dir, random_file)
        file_volume = fslstats(fold_file_path).v.run()
        volumes = np.append(volumes, file_volume[0])

    return volumes, np.mean(volumes)
    


def preprocess_mris(in_dir, out_dir, ref_file):
    """
    in_dir - directory where subject separated scans can be found
    out_dir - directory to contain all preprocessing directories and corresponding files
    ref_file - standard brain template used for registration of brains
    """

    # brain extract and register
    brain_dir = f"{out_dir}/brain_extractions"
    create_dir(brain_dir)
    reg_dir = f"{out_dir}/registered_brains"
    create_dir(reg_dir)
    mat_dir = f"{out_dir}/garbage_matrices"
    create_dir(mat_dir)

    print("extracting brains and registering...")
    bet_and_reg(in_dir, brain_dir, reg_dir, mat_dir, ref_file)

    # extract exact rois and resample them to mean dimensions
    exact_roi_dir = f"{out_dir}/exact_roi"
    create_dir(exact_roi_dir)
    mean_roi_dir = f"{out_dir}/mean_roi_reg_brains"
    create_dir(mean_roi_dir)
    resampled_dir = f"{out_dir}/resampled_reg_brains"
    create_dir(resampled_dir)

    print("extracting rois and resampling...")
    mean_dimensions = extract_exact_roi(reg_dir, exact_roi_dir)
    print(f"mean dimensions: {mean_dimensions}")
    resample_mris(exact_roi_dir, resampled_dir, mean_dimensions)
    shutil.rmtree(exact_roi_dir)

    # straight away extract mean dimensions as ROIs
    print("extracting mean rois...")
    extract_rois(reg_dir, mean_roi_dir, mean_dimensions)

    # starting segmentations
    seg_dir = f"{out_dir}/seg_dir"
    create_dir(seg_dir)

    print("segmenting brains...")
    report_file = open(f"{out_dir}/seg_report.txt", "w")
    perform_segmentations(brain_dir, seg_dir, report_file, is_brain=True)

    seg_files = os.listdir(seg_dir)
    num_brains = remove_seg_waste(brain_dir)
    if len(seg_files) < num_brains:
        print("WARNING - Not all brains were segmented successfully - check the report file for more details")

    # perform volume segmentations
    subcort_vol_dir = f"{out_dir}/SegROIVolumes"
    create_dir(subcort_vol_dir)
    subcort_vol_dir_10 = f"{out_dir}/SegROIVolumes10"
    create_dir(subcort_vol_dir_10)
    subcort_vol_dir_20 = f"{out_dir}/SegROIVolumes20"
    create_dir(subcort_vol_dir_20)
    subcort_vol_dir_30 = f"{out_dir}/SegROIVolumes30"
    create_dir(subcort_vol_dir_30)

    print("extracting segmentation area...")
    mean_dimensions = get_roi_volumes_from_segs(seg_dir, brain_dir, subcort_vol_dir)
    print("extracting segmentation area with 10% increased volume...")
    mean_dimensions_10 = get_roi_volumes_from_segs(seg_dir, brain_dir, subcort_vol_dir_10, growth_factor=1.1)
    print("extracting segmentation area with 20% increased volume...")
    mean_dimensions_20 = get_roi_volumes_from_segs(seg_dir, brain_dir, subcort_vol_dir_20, growth_factor=1.2)
    print("extracting segmentation area with 30% increased volume...")
    mean_dimensions_30 = get_roi_volumes_from_segs(seg_dir, brain_dir, subcort_vol_dir_30, growth_factor=1.3)

    # resampling the extracted sub-volumes
    subcort_dir = f"{out_dir}/SubCortVol"
    create_dir(subcort_dir)
    subcort_dir_10 = f"{out_dir}/SubCortVol10"
    create_dir(subcort_dir_10)
    subcort_dir_20 = f"{out_dir}/SubCortVol20"
    create_dir(subcort_dir_20)
    subcort_dir_30 = f"{out_dir}/SubCortVol30"
    create_dir(subcort_dir_30)

    print("Resampling ROIs...")
    resample_mris(subcort_vol_dir, subcort_dir, mean_dimensions)
    shutil.rmtree(subcort_vol_dir)

    print("Resampling ROIs with 10% increased volume...")
    resample_mris(subcort_vol_dir_10, subcort_dir_10, mean_dimensions_10)
    shutil.rmtree(subcort_vol_dir_10)

    print("Resampling ROIs with 20% increased volume...")
    resample_mris(subcort_vol_dir_20, subcort_dir_20, mean_dimensions_20)
    shutil.rmtree(subcort_vol_dir_20)

    print("Resampling ROIs with 30% increased volume...")
    resample_mris(subcort_vol_dir_30, subcort_dir_30, mean_dimensions_30)
    shutil.rmtree(subcort_vol_dir_30)


def extract_exact_roi_all(in_dir, reg_brain_dir_name, exact_roi_dir_name):
    mean_dimensions_sum = np.array([0.0, 0.0, 0.0])
    folds = os.listdir(in_dir)

    for fold in tqdm(folds, total=len(folds)):
        print(f"fold {fold}")

        fold_dir = os.path.join(in_dir, fold)
        reg_brains_dir = os.path.join(fold_dir, reg_brain_dir_name) 
        exact_roi_out = os.path.join(fold_dir, exact_roi_dir_name)
        create_dir(exact_roi_out)
        mean_dimensions_sum += np.array(extract_exact_roi(reg_brains_dir, exact_roi_out))

    num_folds = len(folds)
    return mean_dimensions_sum / num_folds


def resample_reg_brains_all(folds_dir, exact_roi_dir_name, resampled_brain_out_dir, target_shape):
    folds = os.listdir(folds_dir)

    for fold in tqdm(folds, total=len(folds)):
        print(f"fold {fold}")

        fold_dir = os.path.join(folds_dir, fold)
        exact_brain_roi_dir = os.path.join(fold_dir, exact_roi_dir_name)
        resampled_out = os.path.join(fold_dir, resampled_brain_out_dir)
        resample_mris(exact_brain_roi_dir, resampled_out, target_shape)
        


if __name__ == '__main__':
    fold_dir = ""
    out_exact_roi_dir = ""
    mean_dimensions = extract_exact_roi(fold_dir, out_exact_roi_dir)
    print(f"Mean dimensions for {out_exact_roi_dir}:")
    print(mean_dimensions)

    # then you have to resample the brains with the GLOBAL mean dimensions

