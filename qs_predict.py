# add LDDMM shooting code into path
import sys
sys.path.append('../vectormomentum/Code/Python');
sys.path.append('../library')

from subprocess import call
import argparse
import os.path

#Add deep learning related libraries
from collections import Counter
import torch
import prediction_network
import util
import numpy as np
from skimage import exposure

#Add LDDMM registration related libraries
# pyca modules
import PyCA.Core as ca
import PyCA.Common as common
#import PyCA.Display as display
# vector momentum modules
# others
import logging
import copy
import math

import registration_methods

def predict_parameters():
    predict_param = dict(moving_list=None, target_list=None, output_list = len(moving_list)*['/tcmldrive/hadar/quicksilver/pred_results'], 
    batch_size=64, n_GPU=1, use_correction=True, use_CPU_for_shooting = False,shoot_steps=0,affine_align=True, histeq=False, atlas=../data/atlas/icbm152.nii,
    prediction_saved_model = '../../network_configs/OASIS_predict.pt.tar', correction_saved_model = '../../network_configs/OASIS_correct.pt.tar')
    
    # moving_list/target_list - list of moving/target images files
    # output_list - create a folder for output for each registration
    # TODO: create different folder for each image pair output 
    # for n-GPU>1 - set batch size divisible by the number of GPUs.
    # use_CPU_for_shooting - to save GPU memory
    #shoot_steps - time steps for geodesic shooting. Ignore this option to use the default step size used by the registration model.
    # affine_align - Perform affine registration to align moving and target images to ICBM152 atlas space. Require niftireg
    # atlas - Atlas to use for (affine) pre-registration
    # prediction_saved_model - network parameters for the prediction network
    return predict_param


parser.add_argument('--shoot-steps', type=int, default=0, metavar='N',
                    help='time steps for geodesic shooting. Ignore this option to use the default step size used by the registration model.')
parser.add_argument('--affine-align', action='store_true', default=False,
                    help='Perform affine registration to align moving and target images to ICBM152 atlas space. Require niftireg.')
parser.add_argument('--histeq', action='store_true', default=False,
                    help='Perform histogram equalization to the moving and target images.')



# check validity of input arguments from command line
def check_args(predict_param):
    # number of input images/output prefix consistency check
    n_moving_images = len(predict_param['moving_list'])
    n_target_images = len(predict_param['target_list'])
    n_output_prefix = len(predict_param['output_list'])
    if (n_moving_images != n_target_images):
        print('The number of moving images is not consistent with the number of target images!')
        sys.exit(1)
    elif (n_moving_images != n_output_prefix ):
        print('The number of output prefix is not consistent with the number of input images!')
        sys.exit(1)

    # number of GPU check (positive integers)
    if (predict_param['n_GPU'] <= 0):
        print('Number of GPUs must be positive!')
        sys.exit(1)

    # geodesic shooting step check (positive integers)
    if (predict_param['shoot_steps'] < 0):
        print('Shooting steps (--shoot-steps) is negative. Using model default step.')
#enddef


def create_net(predict_param, network_config):
    net_single = prediction_network.net(network_config['network_feature']).cuda();
    net_single.load_state_dict(network_config['state_dict'])

    if (predict_param['n_GPU'] > 1) :
    # use multiple GPU's
        device_ids=range(0, predict_param['n_GPU'])
        net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
    else:
        net = net_single
    
    net.train()
    return net;
#enddef


def preprocess_image(image_pyca, histeq):
# in this function, a pyca image is loaded as numpy object
# nan values are zeroed 
# image is normalized
# if histeq = True - performs histogram equalization
    image_np = common.AsNPCopy(image_pyca)
    nan_mask = np.isnan(image_np)
    image_np[nan_mask] = 0
    image_np /= np.amax(image_np)

    # perform histogram equalization if needed
    if histeq:
        image_np[image_np != 0] = exposure.equalize_hist(image_np[image_np != 0])

    return image_np


def write_result(result, output_prefix):
    common.SaveITKImage(result['I1'], output_prefix+"I1.mhd")
    common.SaveITKField(result['phiinv'], output_prefix+"phiinv.mhd")
#enddef


#perform deformation prediction
def predict_image(predict_param):
    if (predict_param['use_CPU_for_shooting']):
        mType = ca.MEM_HOST
    else:
        mType = ca.MEM_DEVICE

    # load the prediction network with state dict- predict_network_config
    predict_network_config = torch.load(predict_param['prediction_saved_model'])
    prediction_net = create_net(predict_param, predict_network_config);

    batch_size = predict_param['batch_size']
    patch_size = predict_network_config['patch_size']
    # each element in batch contains 2 3D patches, for moving and for target
    input_batch = torch.zeros(batch_size, 2, patch_size, patch_size, patch_size).cuda()

    # use correction network if required
    if predict_param['use_correction']:
        #  state dict- correction_network_config
        correction_network_config = torch.load(predict_param['correction_saved_model']);
        correction_net = create_net(predict_param, correction_network_config);
    else:
        correction_net = None;

    # start prediction
    for i in range(0, len(predict_param['moving_list'])):

        common.Mkdir_p(os.path.dirname(predict_param['output_list'][i]))
        if (predict_param['affine_align']):
            # Perform affine registration to both moving and target image to the ICBM152 atlas space.
            # Registration is done using Niftireg.
            call(["reg_aladin",
                  "-noSym", "-speeeeed", "-ref", args.atlas ,
                  "-flo", predict_param['moving_list'][i],
                  "-res", predict_param['output_list'][i]+"moving_affine.nii",
                  "-aff", predict_param['output_list'][i]+'moving_affine_transform.txt'])

            call(["reg_aladin",
                  "-noSym", "-speeeeed" ,"-ref", args.atlas ,
                  "-flo", predict_param['target_image'][i],
                  "-res", predict_param['output_list'][i]+"target_affine.nii",
                  "-aff", predict_param['output_list'][i]+'target_affine_transform.txt'])

            moving_image = common.LoadITKImage(predict_param['output_list'][i]+"moving_affine.nii", mType)
            target_image = common.LoadITKImage(predict_param['output_list'][i]+"target_affine.nii", mType)
        else:
            moving_image = common.LoadITKImage(predict_param['moving_list'][i], mType)
            target_image = common.LoadITKImage(predict_param['target_image'][i], mType)

        #preprocessing of the image
        moving_image_np = preprocess_image(moving_image, predict_param['histeq']);
        target_image_np = preprocess_image(target_image, predict_param['histeq']);

        grid = moving_image.grid()
        #moving_image = ca.Image3D(grid, mType)
        #target_image = ca.Image3D(grid, mType)
        moving_image_processed = common.ImFromNPArr(moving_image_np, mType)
        target_image_processed = common.ImFromNPArr(target_image_np, mType)
        moving_image.setGrid(grid)
        target_image.setGrid(grid)

        # Indicating whether we are using the old parameter files for the Neuroimage experiments (use .t7 files from matlab .h5 format)
        # TODO: delete all sections with old experiments .t7 files
        predict_transform_space = False
        # if 'matlab_t7' in predict_network_config:
            # predict_transform_space = True
        # run actual prediction
        prediction_result = util.predict_momentum(moving_image_np, target_image_np, input_batch, batch_size, patch_size, prediction_net, predict_transform_space);
        # this is the predicted momentum of the network
        m0 = prediction_result['image_space']
        #convert to registration space and perform registration
        m0_reg = common.FieldFromNPArr(m0, mType);

        #perform correction
        if (predict_param['use_correction']):
            registration_result = registration_methods.geodesic_shooting(moving_image_processed, target_image_processed, m0_reg, predict_param['shoot_steps'], mType, predict_network_config)
            target_inv_np = common.AsNPCopy(registration_result['I1_inv'])

            correct_transform_space = False
            # if 'matlab_t7' in correction_network_config:
                # correct_transform_space = True
            correction_result = util.predict_momentum(moving_image_np, target_inv_np, input_batch, batch_size, patch_size, correction_net, correct_transform_space);
            m0_correct = correction_result['image_space']
            m0 += m0_correct;
            m0_reg = common.FieldFromNPArr(m0, mType);

        registration_result = registration_methods.geodesic_shooting(moving_image, target_image, m0_reg,predict_param['shoot_steps'], mType, predict_network_config)

        #endif

        write_result(registration_result, predict_param['output_list'][i]);
#enddef



if __name__ == '__main__':
    check_args(predict_param);
    predict_image(predict_param)
