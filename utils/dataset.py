import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from typing import Callable
import cv2
import json
import random
from scipy.ndimage import rotate, zoom, gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter1d

seed_value = 300  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution

def get_nodule_locations(mask_array):
    nonzero_coords = np.argwhere(mask_array > 0)
    if len(nonzero_coords) == 0:
        min_z, min_y, min_x = 0, 0, 0
        max_z, max_y, max_x = mask_array.shape[0], mask_array.shape[1], mask_array.shape[2]
    else:
        min_z, min_y, min_x = np.min(nonzero_coords, axis=0)
        max_z, max_y, max_x = np.max(nonzero_coords, axis=0)     
    min_coords = (min_z, min_y, min_x)
    max_coords = (max_z, max_y, max_x)

    center_z = (min_z + max_z) // 2
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2
    center = (center_z, center_y, center_x)
    return center, min_coords, max_coords


def crop_img_to_nodule(img, mask, target_size, record):
    center, min_coords, max_coords = get_nodule_locations(np.array(mask))
    z, y, x = center
    half_size = [target_size[0] // 2, target_size[1] // 2, target_size[2] // 2]
    if target_size[0] < img.shape[0] or target_size[1] < img.shape[1] or target_size[2] < img.shape[2]:
        z_start = max(0, z - half_size[0])
        z_end = min(img.shape[0], z + half_size[0])
        z_start = max(z_end - target_size[0], 0)
        z_end = z_start + target_size[0]
        y_start = max(0, y - half_size[1])
        y_end = min(img.shape[1], y + half_size[1])
        y_start = max(y_end - target_size[1], 0)
        y_end = y_start + target_size[1]
        x_start = max(0, x - half_size[2])
        x_end = min(img.shape[2], x + half_size[2])
        x_start = max(x_end - target_size[2], 0)
        x_end = x_start + target_size[2]

        img_crop = img[z_start:z_end, y_start:y_end, x_start:x_end]
        mask_crop = mask[z_start:z_end, y_start:y_end, x_start:x_end]
        return img_crop, mask_crop, record
    else:
        return img, mask, record

class JointTransform3D:
    """
    This class is designed for augment the 3D dataset
    input: np.array (z, h, w)
    output: torch.tensor (z, h, w)
    """
    def __init__(self, 
                 patch_size=None,  
                 do_elastic_deform=True, elastic_alpha=(50.0, 200.0), elastic_sigma=(5.0, 15.0),
                 do_rotation=True, rotation_range=(-15, 15), 
                 do_scaling=False, scaling_range=(0.7, 1.4),  
                 do_gamma=True, gamma_range=(0.6, 1.4),  
                 do_noise=True, noise_range=(0.005, 0.03),  
                 do_blur=True, blur_sigma=(0.3, 1.2),  
                 do_brightness=True, brightness_range=(-0.1, 0.1),  
                 do_contrast=True, contrast_range=(0.8, 1.2), 
                 do_mirror=True, mirror_axes=(0, 1, 2),  
                 do_low_resolution=True, zoom_range=(0.5, 1.0),  
                 p_per_sample=0.5,  # Probability of each sample
                 p_el_per_sample=0.2,  # Probability of elastic deformation
                 p_rot_per_sample=0.2,  # Probability of rotation
                 p_scale_per_sample=0.2,  # Scaling probability
                 p_gamma_per_sample=0.2,  # Probability of gamma correction
                 p_noise_per_sample=0.1,  # Probability of adding noise
                 p_blur_per_sample=0.2,  #  Probability of blur
                 p_brightness_per_sample=0.15,  # Probability of brightness adjustment
                 p_contrast_per_sample=0.15,  # Probability of contrast adjustment 
                 p_mirror_per_sample=0.5,  # Probability of mirror
                 p_low_res_per_sample=0.25,  # Probability of low resolution simulation 
                 order_data=3,  
                 order_seg=0,  
                 border_mode_data='constant', 
                 border_mode_seg='constant', 
                 cval_data=0,  
                 cval_seg=0  
                 ):
        self.patch_size = patch_size
        self.do_elastic_deform = do_elastic_deform
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.do_rotation = do_rotation
        self.rotation_range = rotation_range
        self.do_scaling = do_scaling
        self.scaling_range = scaling_range
        self.do_gamma = do_gamma
        self.gamma_range = gamma_range
        self.do_noise = do_noise
        self.noise_range = noise_range
        self.do_blur = do_blur
        self.blur_sigma = blur_sigma
        self.do_brightness = do_brightness
        self.brightness_range = brightness_range
        self.do_contrast = do_contrast
        self.contrast_range = contrast_range
        self.do_mirror = do_mirror
        self.mirror_axes = mirror_axes
        self.do_low_resolution = do_low_resolution
        self.zoom_range = zoom_range
        self.p_per_sample = p_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_gamma_per_sample = p_gamma_per_sample
        self.p_noise_per_sample = p_noise_per_sample
        self.p_blur_per_sample = p_blur_per_sample
        self.p_brightness_per_sample = p_brightness_per_sample
        self.p_contrast_per_sample = p_contrast_per_sample
        self.p_mirror_per_sample = p_mirror_per_sample
        self.p_low_res_per_sample = p_low_res_per_sample
        self.order_data = order_data
        self.order_seg = order_seg
        self.border_mode_data = border_mode_data
        self.border_mode_seg = border_mode_seg
        self.cval_data = cval_data
        self.cval_seg = cval_seg

    def __call__(self, x, y=None):
        """
        :param x: 3D CT, shape (z, h, w)
        :param y: optinal label, shape (z, h, w)
        :return: enhanced x (z, h, w) and corresponding y (z, h, w)
        """
        if random.random() > self.p_per_sample:
            x = torch.from_numpy(x.copy()).float()
            if y is not None:
                y = torch.from_numpy(y.copy()).long() 
                return x, y
            return x

        x = x.copy()
        y = y.copy() if y is not None else None

        if self.patch_size is not None:
            x, y = self.random_crop(x, y)

        if self.do_elastic_deform and random.random() < self.p_el_per_sample:
            x, y = self.elastic_deform(x, y)

        if self.do_rotation and random.random() < self.p_rot_per_sample:
            x, y = self.rotate(x, y)

        if self.do_scaling and random.random() < self.p_scale_per_sample:
            x, y, scale_factor = self.scale(x, y)

        if self.do_mirror and random.random() < self.p_mirror_per_sample and self.mirror_axes:
            x, y = self.mirror(x, y)

        if self.do_low_resolution and random.random() < self.p_low_res_per_sample:
            x = self.simulate_low_resolution(x)

        if self.do_blur and random.random() < self.p_blur_per_sample:
            x = self.gaussian_blur(x)

        if self.do_noise and random.random() < self.p_noise_per_sample:
            x = self.add_gaussian_noise(x)

        if self.do_brightness and random.random() < self.p_brightness_per_sample:
            x = self.adjust_brightness(x)

        if self.do_contrast and random.random() < self.p_contrast_per_sample:
            x = self.adjust_contrast(x)

        if self.do_gamma and random.random() < self.p_gamma_per_sample:
            x = self.gamma_correction(x)

        x = torch.from_numpy(x.copy()).float()
        if y is not None:
            y = torch.from_numpy(y.copy()).long() 
            return x, y
        return x
    

    def call_list(self, x, y):
        """
        :param x: image list, shape (z, h, w)
        :param y: label list, shape (z, h, w)
        :return: enhanced x list (z, h, w) and corresponding y list (z, h, w)
        """
        if random.random() > self.p_per_sample:
            for l in range(len(x)):
                x[l] = torch.from_numpy(x[l].copy()).float()
                y[l] = torch.from_numpy(y[l].copy()).long() 
            return x, y
     

        x = x.copy()
        y = y.copy() 

        if self.patch_size is not None:
            x, y = self.random_crop(x, y)

        if self.do_elastic_deform and random.random() < self.p_el_per_sample:
            x, y = self.elastic_deform(x, y)

        if self.do_rotation and random.random() < self.p_rot_per_sample:
            x, y = self.rotate(x, y)

        if self.do_scaling and random.random() < self.p_scale_per_sample:
            x, y, scale_factor = self.scale(x, y)

        if self.do_mirror and random.random() < self.p_mirror_per_sample and self.mirror_axes:
            x, y = self.mirror(x, y)

        if self.do_low_resolution and random.random() < self.p_low_res_per_sample:
            x = self.simulate_low_resolution(x)

        if self.do_blur and random.random() < self.p_blur_per_sample:
            x = self.gaussian_blur(x)

        if self.do_noise and random.random() < self.p_noise_per_sample:
            x = self.add_gaussian_noise(x)

        if self.do_brightness and random.random() < self.p_brightness_per_sample:
            x = self.adjust_brightness(x)

        if self.do_contrast and random.random() < self.p_contrast_per_sample:
            x = self.adjust_contrast(x)

        if self.do_gamma and random.random() < self.p_gamma_per_sample:
            x = self.gamma_correction(x)

        x = torch.from_numpy(x.copy()).float()
        if y is not None:
            y = torch.from_numpy(y.copy()).long() 
            return x, y
        return x

    def random_crop(self, x, y):
        """ Crop the image """
        z, h, w = x.shape
        z_patch, h_patch, w_patch = self.patch_size
        assert z_patch <= z and h_patch <= h and w_patch <= w, "Patch size larger than image size"

        z_start = random.randint(0, z - z_patch)
        h_start = random.randint(0, h - h_patch)
        w_start = random.randint(0, w - w_patch)

        x_crop = x[z_start:z_start+z_patch, h_start:h_start+h_patch, w_start:w_start+w_patch]
        y_crop = y[z_start:z_start+z_patch, h_start:h_start+h_patch, w_start:w_start+w_patch] if y is not None else None
        return x_crop, y_crop

    def elastic_deform(self, x, y):
        """ Elastic deformation (based on random displacement field) """
        alpha = random.uniform(*self.elastic_alpha)
        sigma = random.uniform(*self.elastic_sigma)
        z, h, w = x.shape

        dx = gaussian_filter1d(np.random.randn(z, h, w), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter1d(np.random.randn(z, h, w), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter1d(np.random.randn(z, h, w), sigma, mode="constant", cval=0) * alpha

        z_coords, h_coords, w_coords = np.mgrid[0:z, 0:h, 0:w]
        indices = (
            np.reshape(z_coords + dz, (-1, 1)),
            np.reshape(h_coords + dy, (-1, 1)),
            np.reshape(w_coords + dx, (-1, 1))
        )

        x_deform = map_coordinates(
            x, indices, order=self.order_data, mode=self.border_mode_data, cval=self.cval_data
        ).reshape(z, h, w)
        
        y_deform = None
        if y is not None:
            y_deform = map_coordinates(
                y, indices, order=self.order_seg, mode=self.border_mode_seg, cval=self.cval_seg
            ).reshape(z, h, w)
        return x_deform, y_deform

    def rotate(self, x, y):
        """ 3D rotation (rotating three axes separately) """
        angles = [random.uniform(*self.rotation_range) for _ in range(3)] 

        x_rot = x
        y_rot = y
        for i, angle in enumerate(angles):
            if angle != 0:
                x_rot = rotate(
                    x_rot, angle, axes=(i, (i+1)%3),  
                    reshape=False, order=self.order_data,
                    mode=self.border_mode_data, cval=self.cval_data
                )
                if y_rot is not None:
                    y_rot = rotate(
                        y_rot, angle, axes=(i, (i+1)%3),
                        reshape=False, order=self.order_seg,
                        mode=self.border_mode_seg, cval=self.cval_seg
                    )
        return x_rot, y_rot

    def scale(self, x, y):
        """ Scale transformation """
        scale = random.uniform(*self.scaling_range)
        z, h, w = x.shape

        x_scale = zoom(
            x, scale, order=self.order_data,
            mode=self.border_mode_data, cval=self.cval_data
        )
  
        x_scale = self.resize_to_original(x_scale, (z, h, w), is_seg=False)

        y_scale = None
        if y is not None:
            y_scale = zoom(
                y, scale, order=self.order_seg,
                mode=self.border_mode_seg, cval=self.cval_seg
            )
            y_scale = self.resize_to_original(y_scale, (z, h, w), is_seg=True)
        return x_scale, y_scale, scale

    def mirror(self, x, y):
        """ Mirror flip """
        for axis in self.mirror_axes:
            if random.random() < 0.5:  
                x = np.flip(x, axis=axis)
                if y is not None:
                    y = np.flip(y, axis=axis)
        return x, y

    def simulate_low_resolution(self, x):
        """ Simulate low resolution images (downsampling first and then upsampling) """
        zoom_factor = random.uniform(*self.zoom_range)
        z, h, w = x.shape
        x_low = zoom(x, zoom_factor, order=0, mode=self.border_mode_data, cval=self.cval_data)
        x_low = zoom(x_low, 1/zoom_factor, order=self.order_data, mode=self.border_mode_data, cval=self.cval_data)
        return self.resize_to_original(x_low, (z, h, w), is_seg=False)

    def gaussian_blur(self, x):
        """ Gaussian blur """
        sigma = random.uniform(*self.blur_sigma)
        return gaussian_filter(x, sigma=sigma, mode=self.border_mode_data, cval=self.cval_data)

    def add_gaussian_noise(self, x):
        """ Add Gaussian noise """
        noise_std = random.uniform(*self.noise_range)
        noise = np.random.normal(0, noise_std, size=x.shape).astype(np.float32)
        return np.clip(x + noise, 0.0, 1.0)

    def adjust_brightness(self, x):
        """ Adjust brightness (add factor) """
        offset = random.uniform(*self.brightness_range)
        return np.clip(x + offset, 0.0, 1.0)  

    def adjust_contrast(self, x):
        """ Adjust contrast (scale after standardization) """
        factor = random.uniform(*self.contrast_range)
        mean = x.mean()
        return np.clip((x - mean) * factor + mean, 0.0, 1.0)

    def gamma_correction(self, x):
        """ Gamma correction (ensuring that the data is non negative) """
        gamma = random.uniform(*self.gamma_range)
        x_non_neg = np.clip(x, 1e-6, 1.0)
        return x_non_neg ** gamma

    def resize_to_original(self, img, original_shape, is_seg=False):
        """ Crop or fill the scaled image back to its original size """
        z_orig, h_orig, w_orig = original_shape
        z, h, w = img.shape

        z_start = max(0, (z - z_orig) // 2)
        h_start = max(0, (h - h_orig) // 2)
        w_start = max(0, (w - w_orig) // 2)

        z_end = z_start + z_orig
        h_end = h_start + h_orig
        w_end = w_start + w_orig

        img_cropped = img[z_start:z_end, h_start:h_end, w_start:w_end]

        if img_cropped.shape != original_shape:
            pad_z = max(0, z_orig - img_cropped.shape[0])
            pad_h = max(0, h_orig - img_cropped.shape[1])
            pad_w = max(0, w_orig - img_cropped.shape[2])
            pad = [(pad_z//2, pad_z - pad_z//2),
                   (pad_h//2, pad_h - pad_h//2),
                   (pad_w//2, pad_w - pad_w//2)]
            cval = self.cval_seg if is_seg else self.cval_data
            img_cropped = np.pad(img_cropped, pad, mode='constant', constant_values=cval)
        return img_cropped

class ImageToImage3D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    """

    def __init__(self, dataset_path: str, split='train', joint_transform: Callable = None, classes=2, img_size=(100, 100, 100), fold=0) -> None:
        
        self.classes = classes
        self.img_size = img_size
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'imgs')
        self.label_path = os.path.join(dataset_path, 'labels')
        self.splits_path = os.path.join(dataset_path, 'splits.json')
        self.data_json_path = os.path.join(dataset_path, 'pulmonary_data_dict.json')
        
        with open(self.data_json_path, 'r', encoding='utf-8') as file:
            self.pulmotary_dict = json.load(file)

        with open(self.splits_path, "r") as f:
            splits_data = json.load(f)

        print(f"Loading Split Fold: {fold+1} / {len(splits_data)}")

        self.all_ids = [
            id_ for id_ in splits_data[fold][split]
            if self.pulmotary_dict.get(id_.split('_aug_')[0], {}).get('nodule_class', 0) != 0
        ]

        self.active_ids = self.all_ids.copy()
        
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
        
        self.dataset = {}
        for c in self.all_ids:
            base_c = c.split('_aug_')[0]
            if base_c not in self.pulmotary_dict: continue

            patient_info = self.pulmotary_dict[base_c]
            self.dataset[c] = {}
            if '_aug_' in c:
                current_ct_name = patient_info['current_ct_name'].split('_')[-1] + '_aug_' + c.split('_aug_')[1]
            else:
                current_ct_name = patient_info['current_ct_name'].split('_')[-1]
            
            self.dataset[c]['current_data_file'] = os.path.join(self.img_path, f"{current_ct_name}.npy")
            self.dataset[c]['current_seg_file'] = os.path.join(self.label_path, f"{current_ct_name}.npy")
            self.dataset[c]['patient_info'] = patient_info
   
    def set_active_ids(self, new_ids_list):
        self.active_ids = new_ids_list

    def __len__(self):
        return len(self.active_ids)

    def __getitem__(self, i):
        id_ = self.active_ids[i]
        entry = self.dataset[id_]

        current_image = np.load(entry['current_data_file'])
        current_mask = np.load(entry['current_seg_file'])
        patient_info = entry['patient_info']
        
        current_ct_info = patient_info.get('current_ct_info', {})
        size_long_val = current_ct_info.get('size_long', -1.0) 
        size_long_tensor = torch.tensor(size_long_val, dtype=torch.float32)

        current_properties = [0] 

        if self.joint_transform:
            current_image, current_mask = self.joint_transform(current_image, current_mask)

        current_image, current_mask, current_properties = crop_img_to_nodule(
            current_image, current_mask, self.img_size, current_properties
        )

        current_mask[current_mask > 0] = 1
        current_mask[current_mask <=0] = 0

        nodule_class = patient_info['nodule_class']
        nodule_class = torch.tensor(nodule_class, dtype=torch.long)
        nodule_class = nodule_class - 1

        return {
            'current_data': current_image.unsqueeze(0), 
            'current_seg': current_mask.unsqueeze(0), 
            'current_properties': current_properties,
            'keys': id_,
            'nodule_class': nodule_class,
            'size_long': size_long_tensor
        }