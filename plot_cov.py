#!/usr/bin/env python3

# Some code in this file was adapted from TLIO
# https://github.com/CathIAS/TLIO
# Here is the code license for TLIO
###################################################################################
# BSD License
#
# For TLIO training and inference software
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###################################################################################

import os
import cv2
import torch
import shutil
import datetime
import numpy as np
from psutil import virtual_memory
from time import time, strftime, mktime

import matplotlib.lines as mlines
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset


from lib.utils import utils
from lib.datasets import bop
from lib.labeling import kp_config
from lib.models.pkpnet import PkpNet
from lib.utils.training_utils import DataParallelWrapper, collate_fn

def plot_sigmas(errs, sigmas, savepath=None,
        ticksize=12, fontsize=14, tickfont="Crimson Text", fontname="Crimson Text"):

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=200)
    plt.sca(axs)
    plt.scatter(errs, sigmas, s=0.3)
    l = mlines.Line2D([-3, 3], [-1, 1], color="r", linestyle="--", linewidth=0.7)
    axs.add_line(l)
    l = mlines.Line2D([3, -3], [-1, 1], color="r", linestyle="--", linewidth=0.7)
    axs.add_line(l)
    plt.xlim((-2, 2))
    plt.ylim((0.0, 1.0))
    plt.setp(axs.get_xticklabels(), fontsize=ticksize, fontname=tickfont)
    plt.setp(axs.get_yticklabels(), fontsize=ticksize, fontname=tickfont)
    plt.grid(True)

    axs.set_ylabel("$\sigma$", fontsize=fontsize, fontname=fontname)
    axs.set_xlabel("$uv$ error", fontsize=fontsize, fontname=fontname)
    plt.tight_layout(pad=0.2)
    if savepath is not None:
        print("Saving plot to", savepath)
        plt.savefig(savepath)
    plt.show()

def main():
    from lib.args import get_args
    args = get_args('eval')

    model = PkpNet(input_res=bop.IMAGE_SIZE)
    chkpt_path = args.checkpoint_path
    assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
    print("=> loading checkpoint '{}'".format(chkpt_path))
    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        print(f"Found CUDA")
        model = model.cuda()
    else:
        print("WARNING: No CUDA found.")
    model.calc_cov = True

    print(f"Loading initial data from {args.data_root}...")
    # Use the train pbr split (every `skip`th image)
    # which has perfect ground truth unlike the YCB real test set.
    # We want perfect ground truth to actually look at the keypoint error
    # NOTE: You CAN NOT use this set here if you trained with it!
    skip = 100
    eval_dataset = bop.BopDataset(args.data_root, "train_pbr", 
            bop_dset=args.dataset, skip=skip, det_type="gt", no_aug=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1,
            shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    model.eval()
    errs = []
    sigmas = []
    num_inlier = 0
    n = 0
    for i, sample in enumerate(eval_loader):
        with torch.no_grad():
            print(f"Running {i+1}/{len(eval_loader)}", end="\r", flush=True)
            image, bboxes = sample["img"], sample["bboxes"] 
            priors, target, kp_mask = sample["priors"], sample["kp_uvs"], sample["kp_masks"]
            # We can concat all but bboxes and priors, since they need to stay in 
            # list form for forward.
            # Images are already stacked.
            target, kp_mask = torch.cat(target), torch.cat(kp_mask)
            if torch.cuda.is_available():
                image, target, kp_mask = [x.cuda() for x in [image, target, kp_mask]]
                bboxes, priors = [b.cuda() for b in bboxes], [p.cuda() for p in priors]

            pred = model(images=image, boxes=bboxes)
            err = (target - pred["uv"])[kp_mask]
            sigma = pred["cov"][kp_mask][...,[0,1],[0,1]].sqrt() # Stdev
                
            chi2 = err.unsqueeze(-2) @ torch.inverse(pred["cov"][kp_mask]) \
                                     @ err.unsqueeze(-1)
            n += chi2.numel()
            # 99% confidence threshold for 2D chi2 distribution
            num_inlier += (chi2 <= 9.210).count_nonzero().item()
            assert sigma.shape == err.shape
            errs.append(err.cpu().numpy().reshape(-1))
            sigmas.append(sigma.cpu().numpy().reshape(-1))
            
            #if i > 30:
            #    break

    print("\n\nDone.")
    print("Percent inside 99% confidence bounds:", float(num_inlier)/n)
    with open(os.path.join(os.path.dirname(args.checkpoint_path), \
            "percent_sigma_inbounds.txt"), 'w') as f:
        print("Percent inside bounds:", float(num_inlier)/n, file=f)
    plot_sigmas(np.concatenate(errs), np.concatenate(sigmas),
            savepath=os.path.join(os.path.dirname(args.checkpoint_path), "sigma_plot.png"))

if __name__ == '__main__':
    main()
