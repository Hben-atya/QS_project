# add LDDMM shooting code into path
import sys
sys.path.append('../../vectormomentum/Code/Python');
from subprocess import call
import argparse
import os.path

#Add LDDMM registration related libraries
# pyca modules
import PyCA.Core as ca
import PyCA.Common as common

import numpy as np
from skimage import exposure
import torch
import torch.datasets


def affine_hist_cf():
    affine_hist_param = dict(input_image_folder="/MLdata/MRIcourse/Data \Bases/supervised_registration/CUMC_examples", output_image_list=None, histeq=False, atlas="MLdata/MRIcourse/Data \Bases/supervised_registration/atlas/icbm152.nii",
    input_labels=None,output_labels=None)
    
    # output images - List of output image directory and file names, seperated by space. Do not save as .nii format as PyCA will flip the image. The recommended format to save images is .mhd
    # output_list - create a folder for output for each registration
    # input/output labels - List of input label maps for the input images, seperated by space.
    # TODO: create different folder for each image pair output 
    # for n-GPU>1 - set batch size divisible by the number of GPUs.
    # use_CPU_for_shooting - to save GPU memory
    #shoot_steps - time steps for geodesic shooting. Ignore this option to use the default step size used by the registration model.
    # affine_align - Perform affine registration to align moving and target images to ICBM152 atlas space. Require niftireg
    # atlas - Atlas to use for (affine) pre-registration
    # prediction_saved_model - network parameters for the prediction network
    return affine_hist_param
    


def check_args(affine_hist_param):
    if (len(affine_hist_param['input_image_list']) != len(affine_hist_param['output_image_list'])):
        print('The number of input images is not consistent with the number of output images!')
        sys.exit(1)
    if ((affine_hist_param['input_labels'] is None) ^ (affine_hist_param['output_labels'] is None)):
        print('The input labels and output labels need to be both defined!')
        sys.exit(1)
    if ((affine_hist_param['input_labels'] is not None) and (len(affine_hist_param['input_labels']) != len(affine_hist_param['output_labels']))):
        print('The number of input labels is not consistent with the number of output labels!')

def affine_transformation(affine_hist_param,dl):
    for image,label in dl:
        call(["reg_aladin",
            "-noSym", "-speeeeed", "-ref", affine_hist_param['atlas'] ,
            "-flo",image,
            "-res", affine_hist_param['output_image_list'][i],
            "-aff", affine_hist_param['output_image_list'][i]+'_affine_transform.txt'])     
        if (label is not None):
            call(["reg_resample",
                "-ref", affine_hist_param['atlas'],
                "-flo",label,
                "-res", affine_hist_param['output_labels'][i],
                "-trans", affine_hist_param['output_image_list'][i]+'_affine_transform.txt',
                "-inter", str(0)]) 

def intensity_normalization_histeq(affine_hist_param):
    for i in range(0, len(affine_hist_param['input_image_list'])):
        image = common.LoadITKImage(affine_hist_param['output_image_list'][i], ca.MEM_HOST)
        grid = image.grid()
        image_np = common.AsNPCopy(image)
        nan_mask = np.isnan(image_np)
        image_np[nan_mask] = 0
        image_np /= np.amax(image_np)

        # perform histogram equalization if needed
        if affine_hist_param['histeq']:
            image_np[image_np != 0] = exposure.equalize_hist(image_np[image_np != 0])
        image_result = common.ImFromNPArr(image_np, ca.MEM_HOST);
        image_result.setGrid(grid)
        common.SaveITKImage(image_result, affine_hist_param['output_image_list'][i])

if __name__ == '__main__':
    affine_hist_param=affine_hist_cf()
    image_dataset = datasets.ImageFolder(affine_hist_param['input_image_folder'])
    dl = torch.utils.data.DataLoader(image_dataset,
                                               batch_size=1,                                              
                                               # shuffle=False,
                                               num_workers = 4)
    print((affine_hist_param['input_labels'] is None) and (affine_hist_param['output_labels'] is None))
    
    affine_transformation(affine_hist_param,dl);
    intensity_normalization_histeq(affine_hist_param)
