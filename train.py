#!/usr/bin/env python3

import os
import cv2
import shutil
import datetime
import numpy as np
from psutil import virtual_memory
from time import time, strftime, mktime

cv2.setNumThreads(0) 

import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

torch.multiprocessing.set_sharing_strategy('file_system')

from lib.utils import utils
from lib.datasets import bop
from lib.labeling import kp_config
from lib.models.pkpnet import PkpNet
from lib.utils.training_utils import DataParallelWrapper, collate_fn

# Perform one epoch of the dataloader, whether train, test, or val
def step_epoch(split, loader, model, epoch, optimizer=None, 
        outdir=None, show_viz=False, do_aneal=True):
    if split == 'train':
        model.train() # switch to train mode
    else:
        model.eval()
    viz_dir = None
    if outdir is not None:
        assert outdir is not None, "Need outdir for test/val"
        viz_dir = os.path.join(outdir, f'viz_{split}_epoch_{epoch}')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
    avg_metric = 0
    n_metric = 0
        
    # Save 8 images for visualization per epoch of data
    skip_viz = max(1, len(loader) // 8)
    t0 = utils.device_time()
    t_sum = 0
    for i, sample in enumerate(loader):
        image, bboxes = sample["img"], sample["bboxes"] 
        priors, target, kp_mask = sample["priors"], sample["kp_uvs"], sample["kp_masks"]
        # We can concat all but bboxes and priors, since they need to stay in 
        # list form for forward.
        # Images are already stacked.
        target, kp_mask = torch.cat(target), torch.cat(kp_mask)
        if torch.cuda.is_available():
            image, target, kp_mask = [x.cuda() for x in [image, target, kp_mask]]
            bboxes, priors = [b.cuda() for b in bboxes], [p.cuda() for p in priors]
        
        if split == 'train':
            optimizer.zero_grad()
            
        # compute pred
        if split != 'train':
            t0 = utils.device_time()
        pred = model(images=image, boxes=bboxes, prior_kp=priors)
        if split != 'train':
            t_sum += utils.device_time() - t0

        uv_loss, var_loss, mask_loss = utils.kp_loss(pred, target, kp_mask, epoch)
        
        # Start weight for var loss at 0 and increase to 1 around epoch 10
        if do_aneal:
            var_lambda = torch.sigmoid(torch.tensor(epoch - 5, dtype=torch.float))
        else:
            var_lambda = 1
        mle_loss = uv_loss + 0.5 * var_lambda * var_loss
        avg_metric += uv_loss.detach().item()
        n_metric += 1

        if split == 'train':
            # Weight the mask loss too
            if do_aneal:
                mask_lambda = torch.sigmoid(torch.tensor(epoch - 10, dtype=torch.float))
            else:
                mask_lambda = 1
            loss = mle_loss + mask_lambda * mask_loss
            loss.backward() # compute gradient and do SGD step
            optimizer.step()
            
            print_freq = 10
            if (i + 1) % print_freq == 0:
                def fmt(i): 
                    reserved_gpu = torch.cuda.memory_reserved(i)
                    total_gpu = torch.cuda.get_device_properties(i).total_memory
                    return f'{100*reserved_gpu / total_gpu:.1f}'
                gpu_usage = [fmt(i) for i in range(torch.cuda.device_count())]
                print(f'Epoch: {epoch} [{i+1}/{len(loader)}] '
                      f'loss_tot={loss:.3f} uv_loss={uv_loss:.3f} '
                      f'var_loss(weight,val)=({var_lambda:.3f},{var_loss:.3f}) '
                      f'mask_loss(weight,val)=({mask_lambda:.3f},{mask_loss:.3f}) '
                      f'sec/it={t_sum/(i+1):.1f} '
                      f'gpu_usage={",".join(gpu_usage)} ')

                # Debug to test val quickly
                #break
            
            # Make sure RAM usage will not freeze the computer
            max_ram_percent = 99
            if virtual_memory().percent > max_ram_percent:
                print(f"RAM usage too high (>{max_ram_percent}%). Exiting.")
                exit()
        else:
            # TODO AUC of ADD with sampled points instead
            print(f'Test: [{i+1}/{len(loader)}] '
                  f'uv_loss={uv_loss:.3f} ({avg_metric/n_metric:.3f} avg) '
                  f'(avg inference time {t_sum/(i+1):.3f} sec)', 
                  end='\r', flush=True)
        
        # Visualize the current result
        #if (split != 'train' or show_viz) and i % skip_viz == 0:
        if i % skip_viz == 0:
            with torch.no_grad():
                image_np = (255 * image[0]).to(torch.uint8).permute(1,2,0).cpu().numpy()
                L = bboxes[0].shape[0]
                kp_pred = pred["uv"][:L].cpu().numpy() # [L K 2]
                cov = None #pred.get("cov", None)
                #if cov is not None:
                #    cov = cov.cpu().numpy()
                kp_indices = sample["kp_masks"][0].numpy() # [L K]
                kp_model_indices = sample["kp_model_masks"][0].numpy() # [L K]
                kp_gt = sample["kp_uvs"][0].numpy() # [L K 2]
                # Convert keypoints to full image plane
                K = sample["K"][0:1].numpy()
                K_kp = sample["K_kps"][0].numpy()
                # [L 3 3] Homography. Transpose so we can right multiply 
                H = (K @ np.linalg.inv(K_kp)).transpose((0,2,1))
                kp_pred = kp_pred @ H[:,:2,:2] + H[:,2:3,:2]
                kp_gt = kp_gt @ H[:,:2,:2] + H[:,2:3,:2]
                kp_prior = sample["prior_uvs"][0].numpy() @ H[:,:2,:2] + H[:,2:3,:2]
                has_prior = sample["has_prior"][0].numpy() 
                # Remake the priors so we don't have to warp the gaussians or anything.
                prior_input = np.zeros((kp_config.num_kp(),*image_np.shape[:2]), dtype=np.float32)
                for k in range(L):
                    if has_prior[k]:
                        prior_input += utils.make_prior_kp_input(kp_prior[k], 
                                kp_model_indices[k], image_np.shape[:2], ndc=False)
                prior_input = np.clip(prior_input, 0, 1)
                rois = np.concatenate((sample["obj_ids"][0].numpy().astype(np.int)[:,None],
                                       (.5+bboxes[0].cpu().numpy()).astype(np.int)), axis=1)
                rgb_viz = utils.make_kp_viz(image_np, kp_pred,kp_indices, kp_gt=kp_gt,
                        bbox_gt=rois, cov=cov, prior=prior_input, ndc=False)

                if show_viz:
                    cv2.imshow("Training example", rgb_viz)
                    cv2.waitKey(1)

                if viz_dir is not None:
                    cv2.imwrite(os.path.join(viz_dir, str(i) + '.png'), rgb_viz)
        
        if split == 'train':
            t1 = utils.device_time()
            t_sum += t1 - t0
            t0 = t1

    print('\n=======================================================')
    return avg_metric / n_metric, viz_dir

def get_output_directory(args):
    ctime = strftime('%m-%d-%Y@%H-%M-%S')
    ext = args.ext
    if len(ext) > 0:
        ext += '_'
    return os.path.join('results', f'pkpnet_{args.dataset}_{args.data_split}_{ext}{ctime}')

def save_checkpoint(state, is_best, epoch, output_directory):
    for cname in [str(epoch), 'latest']:
        checkpoint_filename = os.path.join(output_directory, 
                'checkpoint-' + cname + '.pth.tar')
        torch.save(state, checkpoint_filename)
    if is_best:
        print("Network is best yet! Overwriting previous best...")
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    #if epoch > 0:
    #    prev_checkpoint_filename = os.path.join(output_directory, 
    #            'checkpoint-' + str(epoch-1) + '.pth.tar')
    #    if os.path.exists(prev_checkpoint_filename):
    #        os.remove(prev_checkpoint_filename)

def main():
    from lib.args import get_args
    args = get_args('train')

    # Get the outdir first. If resuming from checkpoint, this will
    # be replaced by the resuming outdir.
    outdir = get_output_directory(args)

    chkpt_path = None
    if args.checkpoint_path is not None:
        print(f"NOTE: Resuming from specified checkpoint_path {args.checkpoint_path}")
        chkpt_path = args.checkpoint_path
        outdir = os.path.dirname(chkpt_path)
    elif not args.no_resume:
        # Resume latest
        # First, find the directories matching the args (besides the datetime)
        model_dirs = []
        outdir_no_datetime = '_'.join(outdir.split('_')[:-1])
        print(f"Searching for previous model paths matching {outdir_no_datetime}")
        for d in os.listdir('results'):
            d = os.path.join('results', d)
            if '_'.join(d.split('_')[:-1]) == outdir_no_datetime:
                model_dirs.append(d)
        if len(model_dirs) > 0:
            # Sort based on the timestamp scalar from the strftimes 
            def strtime2ts(s):
                return mktime(datetime.datetime.strptime(s, '%m-%d-%Y@%H-%M-%S').timetuple())
            model_dirs = sorted(model_dirs, key=lambda s: strtime2ts(s.split('_')[-1]))
            print("NOTE: Found these matches for resuming based on current args: ")
            for d in model_dirs:
                print('\t', d)
            print(f"Looking in most recent {model_dirs[-1]}")
            chkpt_path_ = os.path.join(model_dirs[-1], 'checkpoint-latest.pth.tar')
            if os.path.isfile(chkpt_path_):
                chkpt_path = chkpt_path_
                outdir = model_dirs[-1]
            else:
                print(f"Could not find suitable checkpoint. Training from scratch")
        else:
            print("Could not find matching directory. Training from scratch")
    
    print()
    model = PkpNet(input_res=bop.IMAGE_SIZE, calc_cov=not args.no_network_cov)
    if chkpt_path is not None:
        assert os.path.isfile(chkpt_path), \
                "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        start_epoch = checkpoint['epoch'] + 1
        
        """
        args_curr = args
        args = checkpoint['args'] 
        # Handle code changes that added new args since old checkpoint
        for attr_curr in dir(args_curr):
            if attr_curr not in dir(args):
                # Originally had no augmentations
                if attr_curr == 'no_augmentations':
                    print("NOTE Found checkpoint without no_augmentations arg. "
                          "Setting to old value of True.")
                    setattr(args, attr_curr, True)
                else:
                    setattr(args, attr_curr, getattr(args_curr, attr_curr))
        """

        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        best_val = checkpoint['best_val']
        
        ################# Write in your overrides here for resuming ##########
        # TODO "override_" prefix for args to do this type of thing
        #args.epochs = 1000
        #args.data_root = "/mnt/DATA02/bop/bop_datasets/ycbv"
        #args.workers = 8
        #args.batch_size = 8
        #args.truncate_obj = 36
        ######################################################################

    else:
        if args.pretrain is not None:
            chkpt_path = args.pretrain
            assert os.path.isfile(chkpt_path), \
                    "=> no checkpoint found at '{}'".format(chkpt_path)
            print("=> loading pretrain '{}'".format(chkpt_path))
            checkpoint = torch.load(chkpt_path)
            model.load_state_dict(checkpoint['model'])

        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        start_epoch = 0
        best_val = None
    
        os.makedirs(outdir)


    print(f"Writing results to {outdir}")

    # Write the params to file in human-readable format so we don't forget later
    params_txt = os.path.join(outdir, 'params.txt')
    print("======= Args ================")
    with open(params_txt, 'w') as fp:
        for attr in dir(args):
            # Ignore args attrs we didnt set
            if not attr.startswith('_'):
                val = getattr(args, attr)
                s = f"{attr}: {val}"
                fp.write(s + "\n")
                print(s)
    print("=============================")

    if torch.cuda.is_available():
        ngpus = torch.cuda.device_count()
        print(f"Found CUDA. Training on {ngpus} GPUs")
        model = model.cuda()
        if ngpus > 1:
            model = DataParallelWrapper(model) # for multi-gpu training
    else:
        print("WARNING: No CUDA found.")

    print(f"Loading initial data from {args.data_root}...")
    train_splits = ["train_" + s for s in args.data_split.split("+")]
    if args.dataset == 'tless':
        test_splits = ["test_primesense"]
    else:
        test_splits = ["test"]

    val_dataset = ConcatDataset([bop.BopDataset(args.data_root, split, 
            bop_dset=args.dataset) for split in test_splits])
    train_dataset = ConcatDataset([bop.BopDataset(args.data_root, split, 
            mask_occluded=args.mask_occluded, no_aug=args.no_augmentations, 
            bop_dset=args.dataset, det_type=args.detection_type) for split in train_splits])
    
    val_loader = DataLoader(val_dataset, batch_size=1,
            shuffle=False, num_workers=args.workers, pin_memory=True,
            collate_fn=collate_fn, worker_init_fn=lambda work_id:np.random.seed(666))
    train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, args.truncate_obj))
    
    val_start_epoch = 5
    for epoch in range(start_epoch, args.epochs):
        step_epoch('train', train_loader, model, epoch, optimizer, 
                outdir=outdir, show_viz=args.show_viz, do_aneal=args.pretrain==None)
        # NOTE: This is the test set, so we should not use the "best" network anyways.
        is_best = False
        """
        with torch.no_grad():
            val_err, viz_dir = step_epoch('test', val_loader, model, epoch,
                    outdir=outdir, show_viz=args.show_viz)
        if epoch >= val_start_epoch and (best_val is None or val_err < best_val):
            # Write a small text file with the best info
            with open(os.path.join(outdir, 'best.txt'), 'w') as f:
                f.write(f'epoch={epoch}\nval_err={val_err}\nprev_best={best_val}')
            best_val = val_err
            is_best = True
            # Copy the vizualization to 'viz_best'
            viz_best_dir = os.path.join(outdir, 'viz_best')
            if os.path.exists(viz_best_dir):
                shutil.rmtree(viz_best_dir)
            shutil.copytree(viz_dir, viz_best_dir)
        """
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            'best_val': best_val,
            'optimizer' : optimizer,
        }, is_best, epoch, outdir)

if __name__ == '__main__':
    main()
