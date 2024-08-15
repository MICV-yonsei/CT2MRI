import os
import numpy as np
import nibabel as nib
import time
from scipy.ndimage import zoom


def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.

    :param np.ndarray data: image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: float src_min: (adjusted) offset
    :return: float scale: scale factor
    """

    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale


def scalecrop(data, dst_min, dst_max, src_min, scale):
    """
    Function to crop the intensity ranges to specific min and max values

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: np.ndarray data_new: scaled image data
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new


def rescale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to rescale image intensity values (0-255)

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: np.ndarray data_new: scaled image data
    """
    src_min, scale = getscale(data, dst_min, dst_max, f_low, f_high)
    data_new = scalecrop(data, dst_min, dst_max, src_min, scale)
    return data_new

def intensty_norm(data, dst_min=0., dst_max=1.):
    data -= data.min()
    data = rescale(data, dst_min, dst_max, f_low=0.0, f_high=0.999)
    return data    

def preprocess_all(base, target_size=(192, 192, 160), target_modal=['t2f', 't1n']):
    bitpix = 32
    dtype = np.float32

    error_list = []
    real_start = time.time()
    print(f"total num: {len(list(os.listdir(base)))}")
    for modal in target_modal:
        for pid in os.listdir(base):
            try:
                print(f"{pid}: {modal}")
                start_time = time.time()
                MR_path = os.path.join(base, pid, f'{pid}-{modal}.nii.gz')
                MR_nii = nib.load(MR_path)
                MR_np = MR_nii.get_fdata().astype(dtype)
                MR_np = np.nan_to_num(MR_np)

                # intensity norm 0~1
                MR_np_normed = intensty_norm(MR_np)
                
                # volume fix to 128
                coords = np.argwhere(MR_np_normed > 0.000001)
                start = coords.min(axis=0)
                end = coords.max(axis=0) + 1  # 슬라이스는 종료 인덱스에서 하나 더 크게 설정

                preprocessed_MR_path = os.path.join(base, pid, f'{pid}-{modal}_preprocessed.nii')
                cropped_MR_np_normed = MR_np_normed[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                if any(a > b for a, b in zip(cropped_MR_np_normed.shape, target_size)):
                    print(f"Resize!: {cropped_MR_np_normed.shape}")
                # cropped_MR_np_normed = pad_and_crop_to_target(cropped_MR_np_normed, target_shape=target_size, pad_value=0)
                cropped_MR_np_normed = pad_and_resize_to_target(cropped_MR_np_normed, target_shape=target_size, pad_value=0)

                MR_nii.header['bitpix'] = bitpix
                MR_nii.header['scl_slope'] = 1
                MR_nii.header['scl_inter'] = 0
                print(f"after size: {cropped_MR_np_normed.shape}")
                save_nii = type(MR_nii)(cropped_MR_np_normed, \
                                                affine=MR_nii.affine, \
                                                header=MR_nii.header, \
                                                extra=MR_nii.extra,   \
                                                file_map=MR_nii.file_map)
                nib.save(save_nii, preprocessed_MR_path)

                print(f"Done: {pid}: {modal} / {time.time() - start_time} s")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {pid}: {modal} / {time.time() - start_time} s")
                error_list.append((pid,modal))

        print(f"error_list: {error_list}")
        print(f"Runtime: {time.time() - real_start} s")
        # break


def pad_and_crop_to_target(array, target_shape=(128, 128, 128), pad_value=0):
    """
    Apply symmetric padding and cropping to an array to make it match the target shape.

    Parameters:
    array (np.ndarray): The input 3D array.
    target_shape (tuple): The desired shape (x, y, z).
    pad_value (int or float): The value to use for padding.

    Returns:
    np.ndarray: The padded and/or cropped array.
    """
    # Calculate the padding required for each dimension
    padding = [(max(0, target - size) // 2, max(0, target - size) - max(0, target - size) // 2) for size, target in zip(array.shape, target_shape)]
    
    # Apply padding
    padded_array = np.pad(array, padding, mode='constant', constant_values=pad_value)

    # Calculate the cropping required for each dimension
    cropping = [(max(0, size - target) // 2, max(0, size - target) - max(0, size - target) // 2) for size, target in zip(padded_array.shape, target_shape)]

    # Apply cropping
    cropped_array = padded_array[
        cropping[0][0]:padded_array.shape[0]-cropping[0][1],
        cropping[1][0]:padded_array.shape[1]-cropping[1][1],
        cropping[2][0]:padded_array.shape[2]-cropping[2][1]
    ]

    return cropped_array

def pad_and_resize_to_target(array, target_shape=(128, 128, 128), pad_value=0):
    """
    Apply symmetric padding and resizing to an array to make it match the target shape.

    Parameters:
    array (np.ndarray): The input 3D array.
    target_shape (tuple): The desired shape (x, y, z).
    pad_value (int or float): The value to use for padding.

    Returns:
    np.ndarray: The padded and/or resized array.
    """
    # Initialize the output array
    output_array = np.copy(array)

    # Iterate through each dimension and apply padding or resizing
    for i in range(3):
        # If the current dimension is smaller than the target, apply padding
        if array.shape[i] < target_shape[i]:
            padding = [(0, 0), (0, 0), (0, 0)]
            padding[i] = ((target_shape[i] - array.shape[i]) // 2,
                          (target_shape[i] - array.shape[i] + 1) // 2)
            output_array = np.pad(output_array, padding, mode='constant', constant_values=pad_value)
        # If the current dimension is larger than the target, apply resizing
        elif array.shape[i] > target_shape[i]:
            resize_factor = target_shape[i] / array.shape[i]
            output_array = zoom(output_array, [resize_factor if j == i else 1 for j in range(3)], order=1)

    return output_array

if __name__ == '__main__':
    # base = '~/CT2MRI/Brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    base = '~/CT2MRI/Brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
    target_modal = ['t1n', 't2f', 't1c', 't2c']
    preprocess_all(base, (176, 176, 160), target_modal)