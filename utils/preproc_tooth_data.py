import os
import nibabel as nib
import numpy as np

image_dir = "Images/"
label_dir = "Labels/"
output_root = "Cropped/"
output_image_dir = os.path.join(output_root, "Images")
output_label_dir = os.path.join(output_root, "Labels")
target_shape = (256, 256, 256) #Final max crop size needed (with padding): (242, 224, 184) then round up to (256, 256, 256)
# This can be done by going through all labels and selecting the max bounding box size needed.

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def crop_and_save(image_path, label_path, target_shape):
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    image = image_nii.get_fdata()
    label = label_nii.get_fdata()

    assert image.shape == label.shape, f"Shape mismatch: {image_path}"

    # Get label bbox center
    nonzero = np.array(np.nonzero(label))
    if nonzero.shape[1] == 0:
        print(f"Empty label: {label_path}")
        return

    min_idx = nonzero.min(axis=1)
    max_idx = nonzero.max(axis=1)
    center = ((min_idx + max_idx) // 2).astype(int)

    crop_start, crop_end, pad_before, pad_after = [], [], [], []
    for i in range(3):
        half = target_shape[i] // 2
        start = center[i] - half
        end = center[i] + half
        
        # Clamp within bounds
        if start < 0:
            end += -start
            start = 0
        if end > image.shape[i]:
            diff = end - image.shape[i]
            start = max(0, start - diff)
            end = image.shape[i]

        actual_crop_len = end - start
        pad_total = target_shape[i] - actual_crop_len
        pad_b = pad_total // 2
        pad_a = pad_total - pad_b

        crop_start.append(start)
        crop_end.append(end)
        pad_before.append(pad_b)
        pad_after.append(pad_a)

    slices = tuple(slice(crop_start[i], crop_end[i]) for i in range(3))
    pad_width = tuple((pad_before[i], pad_after[i]) for i in range(3))

    # Crop and pad
    image_crop = np.pad(image[slices], pad_width, mode='constant', constant_values=0)
    label_crop = np.pad(label[slices], pad_width, mode='constant', constant_values=0).astype(np.uint8)

    assert image_crop.shape == target_shape, f"Output shape mismatch: {image_crop.shape}"

    base = os.path.basename(image_path)
    nib.save(nib.Nifti1Image(image_crop, affine=image_nii.affine), os.path.join(output_image_dir, base))
    nib.save(nib.Nifti1Image(label_crop, affine=label_nii.affine), os.path.join(output_label_dir, base))

    print(f"Saved: {base} | Final cropped shape: {image_crop.shape}")
    
    
for fname in os.listdir(image_dir):
    image_path = os.path.join(image_dir, fname)
    label_path = os.path.join(label_dir, fname)
    crop_and_save(image_path, label_path, target_shape)