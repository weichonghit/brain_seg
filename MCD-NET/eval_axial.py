import nibabel as nib
import numpy as np
import time
import sys
import glob
import os.path as op
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms 
from collections import OrderedDict
from os import makedirs
from models.networks import MCD_Net_c

def load_model(args):
    
    # Select the model
    model = MCD_Net_c(args)

    # Put it onto the GPU or CPU
    use_cuda = args['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False

    model.to(device)

    # Set up state dict (remapping of names, if not multiple GPUs/CPUs)
    print("Loading Axial Net from {}".format(args['network_axial_path']))

    model_state = torch.load(args['network_axial_path'], map_location=device)
    new_state_dict = OrderedDict()

    for k, v in model_state["model_state_dict"].items():

        if k[:7] == "module." and not model_parallel:
            new_state_dict[k[7:]] = v

        elif k[:7] != "module." and model_parallel:
            new_state_dict["module." + k] = v

        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    return model
    
def inference_one(model, img_filename, save_as, args):

    print("Reading volume {}".format(img_filename))

    nii_img = nib.load(img_filename)
    orig_data = np.asarray(nii_img.get_fdata(), dtype=np.int32)
    affine_info = nii_img.affine.copy()
    header_info = nii_img.header.copy() 

    test_dataset_axial = OrigDataThickSlices(img_filename, orig_data, transforms=ToTensorTest(), plane='Axial')

    test_data_loader = DataLoader(dataset=test_dataset_axial, batch_size=args['batch_size'], shuffle=False, num_workers=4)

    prediction_probability_axial = torch.zeros((256, args["num_classes"], 256, 256), dtype=torch.float)
    
    print("Axial model loaded. Running...")
    with torch.no_grad():

        start_index = 0
        for batch_idx, sample_batch in enumerate(test_data_loader):
            images_batch = Variable(sample_batch["image"])

            if args['use_cuda']:
                images_batch = images_batch.cuda()

            temp,temp2,temp3 = model(images_batch)
            temp = temp + temp2 + temp3

            prediction_probability_axial[start_index:start_index + temp.shape[0]] = temp.cpu()
            start_index += temp.shape[0]

    prediction_probability_axial = prediction_probability_axial.permute(2, 3, 0, 1)

    prediction_image = torch.argmax(prediction_probability_axial, 3)
    prediction_image = prediction_image.numpy()

    # Saving image
    header_info.set_data_dtype(np.int16)
    mapped_aseg_img = nib.MGHImage(prediction_image, affine_info, header_info)
    nib.save(mapped_aseg_img,save_as)
    print("Saving Segmentation to {}".format(save_as))


# Transformation: the origin plane of ADNI data is sagittal
def transform_coronal(vol, sagittal2coronal=True):
    if sagittal2coronal:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])

# def transform_axial(vol, sagittal2axial=True):
#     if sagittal2axial:
#         return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
#     else:
#         return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])

# Thick slice generator for 7 slice
def get_thick_slices(img_data, slice_thickness=3):
    h, w, d = img_data.shape
    img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),
                                  axis=3)  # edge padding
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint16)
    
    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)

    return img_data_thick

# Class Operator for image loading (orig only)
class OrigDataThickSlices(Dataset):
    def __init__(self, img_filename, orig, plane='Axial', slice_thickness=3, transforms=None):

        try:
            self.img_filename = img_filename
            self.plane = plane
            self.slice_thickness = slice_thickness

            if plane == 'Coronal':
                orig = transform_coronal(orig)
                print('Loading Coronal')

            else:
                # orig = transform_axial(orig)
                print('Loading Axial.')

            # Create Thick Slices
            orig_thick = get_thick_slices(orig, self.slice_thickness)

            # Make 4D
            orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
            self.images = orig_thick

            self.count = self.images.shape[0]

            self.transforms = transforms

            print("Successfully loaded Image from {}".format(img_filename))

        except Exception as e:
            print("Loading failed. {}".format(e))

    def __getitem__(self, index):

        img = self.images[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return {'image': img}

    def __len__(self):
        return self.count

# ToTensor
class ToTensorTest(object):
    def __call__(self, img):

        img = img.astype(np.float32)
        img = img / (img.max()+0.000001) # set to 0 --- 1
        img = img.transpose((2, 0, 1))

        return img


args = {
    'input': '../data_test',
    'output': '../data_test',
    'iname': 'new_orig_nii.nii',
    'oname': 'aseg_mcdnet_cross_357_axial.nii',
    'search_tag': "result*",
    'network_axial_path': './checkpoints/Axial_level_4ckpts/Epoch_50_training_state.pkl',
    'use_cuda': True,
    'batch_size': 32,
    'num_channels': 7, 
    'num_filters': 64, 
    'num_classes': 137
}

if __name__ == "__main__":

    search_path = op.join(args['input'], args['search_tag'])
    subject_directories = glob.glob(search_path)

    # number of subject
    data_set_size = len(subject_directories)
    print("Total Dataset Size is {}".format(data_set_size))
    print('output name:{}'.format(args['oname']))
    count = 0 
    
    model = load_model(args)
    for current_subject in subject_directories:

        start_time = time.time()
        subject = current_subject.split("/")[-1]

        invol = op.join(current_subject, args['iname'])
        save_file_name = op.join(args['output'], subject, args['oname'])
        count += 1
        print("Running MCD-Net on {}".format(subject))
        print('computing:{}/{}'.format(count, data_set_size))

        # Create output subject directory
        sub_dir, out = op.split(save_file_name)

        if not op.exists(sub_dir):
            makedirs(sub_dir)

        # Run network
        
        inference_one(model, invol, save_file_name, args)

        end_time = time.time() - start_time

        print("Total time for segmentation is {:0.4f} seconds.\n".format(end_time))

    sys.exit(0)
