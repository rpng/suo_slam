#!/usr/bin/env python3

import os
import cv2
import argparse
import numpy as np
import pandas as pd

from manual_keypoints import SelectionGui

import lib.utils.utils as utils
from lib.datasets.bop import BopDataset

IMG_WH = 420

# View the choice of symmetry used for training.

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./view_sym_choice.py")
    
    parser.add_argument(
        '--dataset',
        type=str,
        default="ycbv", choices=["ycbv", "tless"],
        help='"ycbv" or "tless"'
    )
    
    parser.add_argument(
        '--ply_file',
        type=str,
        default="/media/nate/Elements/bop/bop_datasets/ycbv/models_fine/obj_000020.ply",
        help='Path to the input PLY mesh file (inside bop dataset directory tree).'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default="/media/nate/Elements/bop/bop_datasets/ycbv/train_real/000003/rgb/000001.png",
        help='Path to the image (inside bop dataset directory tree) to load.'
    )
    
    parser.add_argument(
        '-r', type=int, default=8,
        help='Radius size in pixels of the rendered circles.'
    )

    args = parser.parse_args()
    setattr(args, "kp_config_file", f"./data/{args.dataset}_kp_config.csv")

    obj_id = int(os.path.basename(args.ply_file).split('.')[0].split('obj_')[1])
    view_id = int(os.path.basename(args.image).split('.')[0])
    scene_id = int(os.path.basename(os.path.realpath(os.path.join(args.image,'../..'))))
    data_root = os.path.realpath(os.path.join(args.image,'../../../..'))
    split = os.path.basename(os.path.realpath(os.path.join(args.image,'../../..')))
    print("===========================================================")
    print("DATA_ROOT:", data_root)
    print("SPLIT:", split)
    print(f"scene, view, object = {scene_id}, {view_id}, {obj_id}")
    print("===========================================================")

    gui = SelectionGui(ply_file=args.ply_file, 
            kp_config_file=args.kp_config_file, r=args.r)
    dataset = BopDataset(data_root, split, bop_dset=args.dataset, 
            no_aug=True, det_type="gt", keep_all=True)
    
    T_OtoC = dataset.get_obj_pose(scene_id, view_id, obj_id)
    _, i_best = dataset.pick_symmetry_transform(obj_id-1, T_OtoC)
    # Note that continuous symmetries are discretized
    symmetries = dataset.symmetries["discrete"][obj_id-1]
    n_sym = len(symmetries)
    # Don't make the viz too large
    max_sym = 4
    if n_sym > max_sym:
        inds = list(range(n_sym))
        inds = np.random.choice(inds[:i_best] + inds[i_best+1:],
                size=max_sym-1, replace=False).tolist()
        inds += [i_best]
        print(inds)
        symmetries = [symmetries[ind] for ind in inds]
        n_sym = len(symmetries)
        i_best = n_sym - 1
    data = dataset.get_raw(scene_id, view_id, [obj_id])
    K = data["K"].cpu().numpy()
    image = (255*data["img"].cpu().numpy()).astype(np.uint8).transpose((1,2,0))
    model_kp = data["model_kps"].cpu().numpy()
    model_kp_mask = data["kp_model_masks"].cpu().numpy()

    sq_dim = np.min(image.shape[:2]) # Square dimension for rendered CAD model
    img_combined = np.zeros((image.shape[0], n_sym*image.shape[1]+sq_dim, 3), dtype=np.uint8)
    for i, T_sym in enumerate(symmetries):
        kp_uv = utils.transform_pts(T_OtoC @ T_sym, model_kp) @ K.T
        kp_uv = kp_uv[:,:,:2] / kp_uv[:,:,2:3]
        img_combined[:,i*image.shape[1]:(i+1)*image.shape[1],:] = utils.make_kp_viz(
                image, kp_uv, model_kp_mask, ndc=False, rad=args.r)
        x,y,w,h = dataset.data[scene_id][view_id]["objects"][obj_id]["bbox"]
        s = 15
        x -= s
        y -= s
        w += s
        h += s
        x += i*image.shape[1]
        if i == i_best:
            cv2.rectangle(img_combined, (x,y), (x+w,y+h), [0,255,0], 4)
        else:
            cv2.rectangle(img_combined, (x,y), (x+w,y+h), [0,0,255], 4)
            img_combined = cv2.putText(img_combined, "x", (x+10, y-10), cv2.FONT_HERSHEY_PLAIN, 10, 
                    (0,0,255), 6, cv2.LINE_AA)


    img_rend = gui.inspect_from_file(once=True)
    # Resize to fit better.
    img_rend = cv2.resize(img_rend, (sq_dim, sq_dim))
    x0 = n_sym * image.shape[1]
    img_combined[:img_rend.shape[0], x0:x0+img_rend.shape[1], :] = img_rend
    
    fn = f"./assets/sym_{args.dataset}_{scene_id}_{view_id}_{obj_id}.png"
    print(f"Writing visualization image to {fn}")
    cv2.imwrite(fn, img_combined)
    cv2.imshow(fn, img_combined)
    cv2.waitKey(0)
