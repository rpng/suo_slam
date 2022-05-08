import os
from sys import argv
from getpass import getuser
from argparse import ArgumentParser

def user_defaults(mode):
    user = getuser()
    defaults = {}
    ###########################################################################
    ## Add your username here if you want to change default args
    ###########################################################################
    if user == "nate":
        defaults["workers"] = 4
        if 'tless' in argv:
            defaults["batch_size"] = 16
            defaults["truncate_obj"] = defaults["batch_size"]
        else:
            defaults["batch_size"] = 2
            defaults["truncate_obj"] = 16

    elif user == "rpng":
        defaults["workers"] = 8
        if 'tless' in argv:
            # NOTE: Need to change if using sim data. This is for the primesense
            # data which has 1 object per image.
            defaults["batch_size"] = 56
            defaults["truncate_obj"] = defaults["batch_size"]
        else:
            defaults["batch_size"] = 12
            defaults["truncate_obj"] = 64

    elif user == "zuox":
        defaults["workers"] = 12 # 8
        if 'tless' in argv:
            defaults["batch_size"] = 128
            defaults["truncate_obj"] = defaults["batch_size"]
        else:
            defaults["batch_size"] = 24 # 6
            defaults["truncate_obj"] = 128 # 32

    # User-independent args
    defaults["dataset"] = "ycbv"
    if mode == 'train':
        defaults["checkpoint_path"] = None
        defaults["detection_type"] = "gt+noise"
    else:
        defaults["checkpoint_path"] = "results/pkpnet_09-28-2021@15-54-39/model_best.pth.tar"
        defaults["detection_type"] = "saved"

    return defaults

def get_args(mode='train'):
    assert mode in ['train', 'eval']
    parser = ArgumentParser(description=('Evaluate' if mode=='eval' else 'Train') + ' PkpNet')

    defaults = user_defaults(mode)
    parser.add_argument('--checkpoint_path', '-c', default=defaults["checkpoint_path"], 
            help=f'Path to the checkpoint file to load (resume for training or run for eval). '
            + f'(default={defaults["checkpoint_path"]})')
    parser.add_argument('--dataset', '-d', default=defaults["dataset"], 
            choices=['ycbv', 'tless'], help=f'Dataset type (default={defaults["dataset"]}).')
    parser.add_argument('--no_network_cov', '-u', action='store_true',
            help='If set, ignore the uncertainty predicted by the net and '
            + 'skip MLE loss if training.')
    parser.add_argument('--show_viz', action='store_true',
            help='If set, visualize the results while training or testing.')
    det_choices = ['gt','gt+noise']
    if mode != "train":
        det_choices.append('saved')
    parser.add_argument('--detection_type', '-t', default=defaults["detection_type"], 
            choices=det_choices, 
            help=f'Type of detection to feed network (default={defaults["detection_type"]}).')
    
    default_split = 'primesense' if 'tless' in argv else 'real+synt'

    # Mode-specific args
    if mode == 'train':
        parser.add_argument('--workers', '-j', type=int, default=defaults["workers"], 
                help=f'Maximum number of workers in dataloader (some eval may use less).')
        parser.add_argument('--batch_size', '-b', type=int, default=defaults["batch_size"], 
                help=f'Maximum batch_size for machine (some eval may use less). '
                + 'Note that multiple objects are loaded for each batch, so the '
                + 'true batch size run through the keypoint model will be larger than this.')
        parser.add_argument('--epochs', type=int, default=1000 if 'tless' in argv else 30, 
                help=f'Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=1e-3, 
                help=f'Learning rate.')
        parser.add_argument('--ext', default="", 
                help=f'Extension to place on the directory name for organizational purposes. '
                      'Also, if a directory matching ext is already found, then resume from the '
                      'most recent one (unless no_resume is set)')
        parser.add_argument('--no_resume', action='store_true',
                help='The training code typically looks for the most recent directory matching '
                     'the current args for resuming. If no_resume is set, then train from scrath.')
        parser.add_argument('--pretrain', default=None, 
                help=f'Path to the checkpoint file to use for a pretrained '
                      'network without resuming.')
        parser.add_argument('--data_split', default=default_split, 
                help=f'"+"-separated list of the training splits to use. '
                'Can be any combination of "real", "synt", and "pbr". See the BOP '
                f'website for more details on what these are (default={default_split}).')
        parser.add_argument('--truncate_obj', type=int, default=defaults["truncate_obj"], 
                help=f'Truncate a batch to this many objects so you can leave the batch size '
                + 'larger but avoid memory overflow. Warnings will be printed if more than '
                + 'a few objects are truncated.')
        parser.add_argument('--mask_occluded', action='store_true',
                help='If set, train the network to only detect visible keypoints.')
        parser.add_argument('--no_augmentations', action='store_true',
                help='If set, skip the training data augmentations.')
    else:
        parser.add_argument('--nviews', type=int, default=-1, 
                help=f'Number of views to eval with. --nviews=1 returns just the single-view '
                'PnP results (with covariance-weighted refinement if using covariance), '
                '--nviews=N for some N>1 (typically small like 5 or 8) will perform a SfM type '
                'evaluation. In this case, the SfM problem will be run separately for each '
                'view with the current view in the dataset as the global frame in the problem. '
                'This will be rather slow to process because of this. Set to -1 to eval all '
                'views sequentially in SLAM fashion. (default=-1).')
        # TODO I feel like there's a better way to do the viz args, like some sort of list
        # of things you want to viz.
        parser.add_argument('--no_viz', action='store_true',
                help='If set, save some time by skipping visualizations altogether. '
                'Visualizations typically take 200ms or so per image, so it\'s alot.')
        parser.add_argument('--viz_cov', action='store_true',
                help='If set, visualize the covariance in the images (may be visually busy).')
        parser.add_argument('--do_viz_extra', action='store_true',
                help='If set, add extra visualizations of each individual object.')
        parser.add_argument('--no_prior_det', '-p', action='store_true',
                help='If set, skip the prior detections which disambiguate multiview results.')
        parser.add_argument('--debug_gt_kp', action='store_true',
                help='If set, debug the evaluation with the GT keypoints instead of estimated. '
                'Note that there is some noise added to the GT keypoints to keep the information '
                'matrix of the PnP and BA well-conditioned, so the result is not "perfect"')
        parser.add_argument('--gt_cam_pose', action='store_true',
                help='If set, give the SLAM system the GT camera poses.')
        parser.add_argument('--debug_saved_only', action='store_true',
                help='If set, just eval the saved detections to debug the AUC of ADD(-S).')
        parser.add_argument('--give_all_prior', action='store_true',
                help='If set, consider all objects the prior detection regardless of '
                'symmetric or not. If gt_cam_pose is not set, then other methods will '
                'try and estimate the camera pose from the bboxes or a constant velocity model.')
    args = parser.parse_args()
    # Make sure all data downloaded is in ./data (can be a symlink)
    setattr(args, "data_root", os.path.join(os.getcwd(), 'data/bop_datasets/', args.dataset))
    return args

