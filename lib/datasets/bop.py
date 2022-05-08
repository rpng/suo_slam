import os
import cv2
import sys
import json
import time
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict

cv2.setNumThreads(0) 

import torch
from torch.utils.data import Dataset

from ..utils import utils
from ..labeling import kp_config
from ..datasets import augmentations as aug

IMAGE_SIZE = (256, 256)

# Image file utils
IMG_EXTENSIONS = ['.jpg','.jpeg','.JPEG','.png']
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class BopDataset(Dataset):

    def __init__(self, data_root, split, bop_dset="ycbv", map_by="view", 
            mask_occluded=False, ignore_symmetry=False, no_aug=False,
            det_type="gt", keep_all=False):
        """ 
        :param data_root: string of full path to the root of the desired BOP dataset
                          (i.e. should contain models dir, train_*, test_*, etc
        :param split: The string of the split should match the directory under data_root
                      that you want to get data from.
        :param bop_dset: Whic bop dataset this is. 
        :param map_by: "view", "obj", or "obj_<obj_id>". If "view", the dataloader 
                       len is the number of views, and each sample is all the objects in view.
                       if "obj", the len is the total number of objects, and each sample
                       is just one object in view. If "obj_<obj_id>" then its the same as before
                       but for only the object with obj_id (i.e. single-object model).
        :param mask_occluded: If set, include occlusion in the keypoint masks such that keypoints
                              that are masked out are either not preent for the object or occluded.
        :param ignore_symmetry: If set, just use the GT pose and ignore the symmetry.
        :param no_aug: If set, don't do the data augmentations on the training data. Test
                       data has no augmentations anyways.
        """
        t0 = time.time()
        self.data_root = data_root
        self.split = split
        assert bop_dset in ["ycbv", "tless"]
        self.bop_dset = bop_dset
        self.keep_all = keep_all

        assert map_by=="view" or "obj" in map_by
        self.single_obj = None
        if "obj_" in map_by:
            self.single_obj = int(map_by.split("_")[1])
        self.map_by = map_by
        self.mask_occluded = mask_occluded
        self.ignore_symmetry = ignore_symmetry
        self.kp_config_file = f"./kp_configs/{self.bop_dset}_kp_config.csv"
        self.kp_path = os.path.join(data_root, "kp_info")
        #self.bop_root = os.path.realpath(os.path.join(data_root, "../.."))
        self.bop_root = os.path.realpath(os.path.join(data_root, ".."))
        if self.should_load_bg_images():
            self.bg_images_dir = os.path.join(self.bop_root, "VOCdevkit/VOC2012/JPEGImages")
            print(f"Loading background images from {self.bg_images_dir}...")
            if not os.path.exists(self.bg_images_dir):
                assert False, f"Background image dir {self.bg_images_dir} does not exist. " + \
                        "Please download and extract " + \
                        "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar " + \
                        "or point this path to any other flat directory of background images."

            # Load images for background of synthetic data
            self.load_bg_images()
        
        self.no_aug = no_aug or 'train' not in split
        if self.no_aug:
            self.img_aug = []
        else:
            self.img_aug = [
                aug.NpScaleAndRotate(),
                aug.PillowBlur(p=0.4, factor_interval=(1, 3)),
                aug.PillowSharpness(p=0.3, factor_interval=(0., 50.)),
                aug.PillowContrast(p=0.3, factor_interval=(0.2, 50.)),
                aug.PillowBrightness(p=0.5, factor_interval=(0.1, 6.0)),
                aug.PillowColor(p=0.3, factor_interval=(0., 20.)),
            ]
    
        assert det_type in ["gt", "gt+noise"]
        self.det_type = det_type

        # Load the kp configs for each object (i.e. what kps they have)
        self.load_kp_config()

        # Load the 3D CAD model keypoints
        self.load_kp()
        
        # Set of transformations under which the object looks the same rendered.
        self.load_object_symmetries()

        curr_root = os.path.join(data_root, split)
        self.curr_root = curr_root
        print(f"Searching for {split} data under {curr_root}")

        min_visib_fract = -1
        if "train" in split or self.bop_dset == "tless":
            min_visib_fract = 0.1 # Ignore completely hidden objects
            
        # Test set is only for some keyframes (or specific targets in case of tless)
        keyframes = None
        self.targets = None
        self.targets_filename = None
        if "test" in split:
            if bop_dset == "ycbv":
                with open(os.path.join(data_root, "keyframe.txt"), 'r') as f:
                    keyframes = f.read().split("\n")[:-1]
            elif bop_dset == "tless":
                self.targets_filename = os.path.join(data_root, "all_target_tless.json")
                with open(self.targets_filename, 'r') as f:
                    targets_list = json.load(f)
                # Remap the targets for easier access
                self.targets = defaultdict(dict)
                for target in targets_list:
                    assert target["inst_count"] == 1
                    if target["im_id"] not in self.targets[target["scene_id"]].keys():
                        self.targets[target["scene_id"]][target["im_id"]] = []
                    self.targets[target["scene_id"]][target["im_id"]].append(target["obj_id"])
            else:
                assert False

        # Map scenes by int of scene_id, view by view_id, the object by object_id.
        # Then, the arrays in object_index_map will map index in __getitem__ back
        # to this main nested dict.
        self.data = {}

        # One entry in each key for each object in each image. This will determine the 
        # size of this dataset by __len__, and points to indices in self.data
        # needed when training or evaluating objects independently.
        self.object_index_map = {
            "scene_ids": [],
            "view_ids": [],
            "obj_ids": []
        }
        
        # Map by frame instead.
        # To get obj_ids just access self.data[scene_ids[i]][view_ids[i]].keys()
        self.view_index_map = {
            "scene_ids": [],
            "view_ids": [],
        }

        # Loop over scenes
        min_bbwh = 1000
        frame_count = 0
        for scene_id_str in sorted(os.listdir(curr_root)):
            scene_dir = os.path.join(curr_root, scene_id_str)
            if not os.path.isdir(scene_dir):
                continue
            scene_id = int(scene_id_str)
            scene = {}
            # Camera poses and intrinsics
            with open(os.path.join(scene_dir, 'scene_camera.json'), 'r') as f:
                cam_infos = json.load(f)
            # bboxes, etc
            with open(os.path.join(scene_dir, 'scene_gt_info.json'), 'r') as f:
                gt_infos = json.load(f)
            # Object poses
            with open(os.path.join(scene_dir, 'scene_gt.json'), 'r') as f:
                gt_poses = json.load(f)
            # Now loop through images
            for view_id_str in cam_infos.keys():
                view_id = int(view_id_str)
                # Skip every 5 frames on ycbv train real
                keep_kf = True
                obj_to_keep = None
                if self.bop_dset == "ycbv" and self.split == "train_real":
                    keep_kf = frame_count % 5 == 0
                frame_count += 1
                if keyframes is not None: # Check if this image is a keyframe
                    keep_kf = False
                    for kf in keyframes:
                        kf_scene_id, kf_view_id = kf.split('/')
                        kf_scene_id, kf_view_id = int(kf_scene_id), int(kf_view_id)
                        if kf_scene_id == scene_id and kf_view_id == view_id:
                            keep_kf = True
                            # No need to check this one again... 
                            # Although probs doesn't save much speed.
                            keyframes.remove(kf) 
                            break
                elif self.targets is not None:
                    keep_kf = scene_id in self.targets.keys() \
                            and view_id in self.targets[scene_id].keys()
                    if keep_kf:
                        obj_to_keep = self.targets[scene_id][view_id]

                if self.single_obj is not None:
                    obj_to_keep = [self.single_obj]

                if keep_kf:
                    frame = {
                        "objects": {}, # Map by obj_id
                        "K": np.array(cam_infos[view_id_str]["cam_K"],dtype=np.float64)
                                .reshape(3,3),
                        "depth_scale": cam_infos[view_id_str]["depth_scale"],
                    } 
                    if "cam_R_w2c" in cam_infos[view_id_str].keys():
                        R = np.array(cam_infos[view_id_str]["cam_R_w2c"],
                                dtype=np.float64).reshape(3,3)
                        t = np.array(cam_infos[view_id_str]["cam_t_w2c"],
                                dtype=np.float64).reshape(3,1)
                        frame["cam_pose"] = np.concatenate((R,t), axis=-1)

                    # Now index all the GT objects
                    for obj_idx, obj_gt in enumerate(gt_poses[view_id_str]):
                        obj_info = gt_infos[view_id_str][obj_idx]
                        if obj_info["visib_fract"] >= min_visib_fract:
                            obj_id = obj_gt["obj_id"]
                            if obj_to_keep is None or obj_id in obj_to_keep:
                                self.object_index_map["scene_ids"].append(scene_id)
                                self.object_index_map["view_ids"].append(view_id)
                                self.object_index_map["obj_ids"].append(obj_id)
                                R = np.array(obj_gt["cam_R_m2c"],dtype=np.float64).reshape(3,3)
                                t = np.array(obj_gt["cam_t_m2c"],dtype=np.float64).reshape(3,1)
                                pose = np.concatenate((R,t), axis=-1)
                                scene_root = os.path.join(self.curr_root, f"{scene_id:06d}")
                                mask_path = os.path.join(scene_root, "mask_visib", 
                                        view_id_str.zfill(6)+f"_{obj_idx:06d}.png")
                                frame["objects"][obj_id] = {
                                    "mask_path": mask_path,
                                    "bbox": obj_info["bbox_visib"],
                                    "pose": pose
                                }
                                min_bbwh = min(min_bbwh, obj_info["bbox_visib"][2])
                                min_bbwh = min(min_bbwh, obj_info["bbox_visib"][3])

                    if len(frame["objects"]) > 0:
                        scene[view_id] = frame
                        self.view_index_map["scene_ids"].append(scene_id)
                        self.view_index_map["view_ids"].append(view_id)

            # For single object, some scenes may not have the object
            if len(scene) > 0:
                self.data[scene_id] = scene

        t_init = time.time() - t0
        print(f"Found {len(self.data)} scenes with " +
                f"{len(self.view_index_map['scene_ids'])} {split} " +
                f"images ({frame_count} total images) " +
                f"with {len(self.object_index_map['obj_ids'])} " +
                f"gt object detections ({t_init:.3f} sec).")
        #print(f"Min bbox wh is {min_bbwh}")

    def is_target(self, scene_id, view_id, obj_id):
        return self.targets is None or \
                obj_id in self.targets.get(scene_id,{}).get(view_id,[])

    def should_load_bg_images(self):
        return "synt" in self.split or self.bop_dset=="tless" and self.split=="train_primesense"

    def load_bg_images(self):
        self.bg_image_files = []
        for fname in os.listdir(self.bg_images_dir):
            if is_image_file(fname):
                self.bg_image_files.append(os.path.join(self.bg_images_dir, fname))

    def load_kp_config(self):
        data = pd.read_csv(self.kp_config_file)
        self.kp_map_per_object = []
        self.kp_list_per_object = []
        for object_idx in range(data.shape[0]):
            kp_config_df = data.iloc(0)[object_idx]
            kp_map = kp_config.load_kp_config(data, object_idx+1)
            kp_list = []
            for k in kp_config.kp_list:
                if k in kp_map.keys():
                    kp_list.append(k)                
            self.kp_map_per_object.append(kp_map)
            self.kp_list_per_object.append(kp_list)
        
    def num_obj(self):
        return len(self.kp_map_per_object)

    def load_kp(self):
        self.gt_kp = [] # One set of kp per object instance
        for object_idx in range(self.num_obj()):
            object_id = str(object_idx+1)
            kp_file = os.path.join(self.kp_path, "obj_" + object_id.zfill(6) + "_kp_info.json")
            #print(f"Loading keypoints for object {object_id} from {kp_file}")
            if not os.path.exists(kp_file):
                assert False, f"No keypoint file {kp_file} found. " + \
                        "Please run `./manual_keypoints.py` for this object."
            with open(kp_file, 'r') as f:
                kp_data = json.load(f)
            kp_list = self.kp_list_per_object[object_idx]
            kp_avg = np.empty((len(kp_list), 3))
            kp_cov = np.empty((len(kp_list), 3, 3))
            for i, kp_name in enumerate(kp_list):
                kp_avg[i] = kp_data["keypoints"][kp_name]["pos_mean"]
                kp_cov[i] = np.array(kp_data["keypoints"][kp_name]["pos_cov"]).reshape(3,3)
            self.gt_kp.append({
                "kp_avg": kp_avg,
                "kp_cov": kp_cov,
                "view_pose": np.array(kp_data["view_pose"]).reshape(4,4),
            })
    
    def load_object_symmetries(self):
        self.obj_sym = []
        if self.bop_dset == 'ycbv':
            # Original YCBV models compat with BOP gt data. Download with download_data.py
            models = 'models_bop-compat' 
        elif self.bop_dset == 'tless':
            models = 'models_cad' 
        else:
            assert False
        with open(os.path.join(self.data_root, models, 'models_info.json'), 'r') as f:
            model_info = json.load(f)
        self.symmetries = {"discrete": [], "continuous": []}
        for object_idx in range(self.num_obj()):
            object_id = str(object_idx+1)
            info_i = model_info[object_id]
            # Identity is ommitted in the file https://github.com/thodan/bop_toolkit/issues/50
            sym_dis = [np.eye(4)]
            if "symmetries_discrete" in info_i.keys():
                for sym in info_i["symmetries_discrete"]:
                    sym_dis.append(np.array(sym, dtype=np.float64).reshape(4,4))
            sym_con = []
            n_discretize = 64 # Discretize into this amount
            if "symmetries_continuous" in info_i.keys():
                for sym in info_i["symmetries_continuous"]:
                    axis = np.array(sym["axis"], dtype=np.float64).reshape(3)
                    offset = np.array(sym["offset"], dtype=np.float64).reshape(3)
                    sym_con.append({
                        "axis": axis,
                        "offset": offset,
                    })
                    # Discretize and add to symmetries_discrete
                    assert np.allclose(offset, 0)
                    assert axis.sum() == 1
                    for n in range(n_discretize):
                        euler = axis * 360 * n / n_discretize
                        T_sym = np.eye(4)
                        T_sym[:3,:3] = utils.euler2R(euler)
                        sym_dis.append(T_sym)
            #print(f"Object {object_idx+1} has {len(sym_dis)} discrete symmetries ", end="")
            #if len(sym_con) > 0:
            #    print(f"({n_discretize} are discretized from continuous)", end="")
            #print()
            self.symmetries["discrete"].append(sym_dis)
            self.symmetries["continuous"].append(sym_con)

    # If there are symmetries, pick the symmetry T_s that minimizes the distance
    # between 3D keypoints projected into the camera with T_OtoC*T_s and T_VtoC,
    # where T_VtoC is the object's view pose.
    def pick_symmetry_transform(self, object_idx, T_OtoC, random=False):
        symmetries = self.symmetries["discrete"][object_idx]
        assert len(symmetries) >= 1
        if len(symmetries) == 1: # All symmetries have been appended with identity
            return T_OtoC, 0
        if random:
            i = np.random.choice(len(symmetries))
            return T_OtoC @ symmetries[i], i
        else:
            # Transform points and mean subtract
            def tpms(T, p):
                pt = utils.transform_pts(T, p)
                return pt - np.mean(pt, axis=0)[None,:]
            T_VtoC = self.gt_kp[object_idx]["view_pose"] # [4,4]
            p = self.gt_kp[object_idx]["kp_avg"] # [N,3]
            smallest_dist = None
            T_OtoC_best = np.copy(T_OtoC)
            i_best = -1
            for i, T_sym in enumerate(symmetries):
                T_OtoC_sym = T_OtoC @ T_sym
                dist = np.linalg.norm(tpms(T_OtoC_sym,p) - tpms(T_VtoC,p), axis=-1)
                assert len(dist.shape) == 1
                dist = np.mean(dist)
                if smallest_dist is None or dist < smallest_dist:
                    T_OtoC_best = T_OtoC_sym
                    smallest_dist = dist
                    i_best = i
            return T_OtoC_best, i_best

    def __len__(self):
        if self.map_by == "view":
            return len(self.view_index_map["scene_ids"]) # To map by frame
        else:
            return len(self.object_index_map["scene_ids"]) # To map by object
 
    # May return None if not avail
    # If view_id  is -1, get the first view.
    def get_cam_pose(self, scene_id, view_id=-1):
        if view_id < 0:
            view_id = min(self.data[scene_id].keys())
        return self.data[scene_id][view_id].get("cam_pose")
    
    def get_obj_pose(self, scene_id, view_id, obj_id):
        return self.data[scene_id][view_id]["objects"][obj_id]["pose"]

    # For looping over scene_id and view_id
    def scene_ids(self):
        return list(self.data.keys())
    
    # For looping over scene_id and view_id
    # Example:
    # for scene_id in dataset.scene_ids():
    #    for view_id in dataset.view_ids(scene_id):
    #        ...
    def view_ids(self, scene_id):
        return list(self.data[scene_id].keys())
    
    def obj_ids(self, scene_id, view_id):
        return list(self.data[scene_id][view_id]["objects"].keys())

    def __getitem__(self, index):
        if self.map_by == "view":
            return self.get_all_obj(self.view_index_map["scene_ids"][index],
                                    self.view_index_map["view_ids"][index])
        else:
            return self.get_raw(self.object_index_map["scene_ids"][index],
                                self.object_index_map["view_ids"][index],
                               [self.object_index_map["obj_ids"][index]])
    
    # Get all the gt info for this frame
    def get_all_obj(self, scene_id, view_id):
        obj_ids = [k for k in self.data[scene_id][view_id]["objects"]]
        return self.get_raw(scene_id, view_id, obj_ids)

    def read_img(self, scene_id, view_id):
        img_ext = ".jpg" if "pbr" in self.split else ".png" 
        view_id_str = f"{view_id:06d}"
        scene_root = os.path.join(self.curr_root, f"{scene_id:06d}")
        image_path = os.path.join(scene_root, "rgb", view_id_str+img_ext)

        img0 = cv2.imread(image_path)
        assert img0 is not None, f"Empty image {image_path}"
        assert img0.size > 0, f"Empty image {image_path}"
        #assert img0.shape == (480, 640, 3)
        assert img0.dtype == np.uint8
        return img0

    def read_depth(self, scene_id, view_id):
        img_ext = ".jpg" if "pbr" in self.split else ".png" 
        view_id_str = f"{view_id:06d}"
        scene_root = os.path.join(self.curr_root, f"{scene_id:06d}")
        depth_path = os.path.join(scene_root, "depth", view_id_str+".png")
        depth0 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        assert depth0 is not None, f"Empty depth image {depth_path}"
        assert depth0.size > 0, f"Empty depth image {depth_path}"
        assert depth0.dtype == np.uint16
        # Get depth in mm
        depth0 = np.squeeze(depth0.astype(np.float32)) \
                * self.data[scene_id][view_id]["depth_scale"] 
        return depth0

    def read_mask(self, scene_id, view_id, obj_id):
        obj_info = self.data[scene_id][view_id]["objects"][obj_id]
        mask_path = obj_info["mask_path"]
        mask0 = np.squeeze(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        assert mask0 is not None, f"Empty image {mask_path}"
        assert mask0.size > 0, f"Empty image {mask_path}"
        assert mask0.dtype == np.uint8
        return mask0

    # scene_id, view_id are int, obj_ids is list of ints.
    def get_raw(self, scene_id, view_id, obj_ids):
        img0 = self.read_img(scene_id, view_id)

        # Data to return. torch can still stack batches of dict
        # by looping through all the elements.
        K = self.data[scene_id][view_id]["K"]
        
        depth0 = None
        if self.mask_occluded or self.should_load_bg_images() \
                and not (self.bop_dset=="tless" and self.split=="train_primesense"):
            depth0 = self.read_depth(scene_id, view_id)
            assert depth0.shape[:2] == img0.shape[:2]

        paste_imgs = []
        if self.should_load_bg_images():
            if self.bop_dset=="tless" and self.split=="train_primesense":
                # Load the mask for this one. Only single object per image, and no
                # other easy way to reliably mask the background.
                assert len(obj_ids) == 1
                mask0 = self.read_mask(scene_id, view_id, obj_ids[0])
                assert mask0.shape == img0.shape[:2]
                img0_bg_mask = mask0 != 255
        
                # Load some other objects randomly to paste in the image
                # to create occlusion
                num_paste = np.random.randint(0, 3)
                for paste in range(num_paste):
                    obj_map_idx = np.random.randint(0, len(self.object_index_map["obj_ids"]))
                    scene_id_paste = self.object_index_map["scene_ids"][obj_map_idx]
                    view_id_paste = self.object_index_map["view_ids"][obj_map_idx]
                    obj_id_paste = self.object_index_map["obj_ids"][obj_map_idx]

                    img_paste = self.read_img(scene_id_paste, view_id_paste)
                    mask_paste = self.read_mask(scene_id_paste, view_id_paste, obj_id_paste)
                    x,y,w,h = self.data[scene_id_paste][view_id_paste]["objects"][obj_id_paste]["bbox"]
                    #img_paste[mask_paste==0] = 0
                    img_paste = img_paste[y:y+h, x:x+w]
                    paste_imgs.append( (img_paste, mask_paste[y:y+h,x:x+w]==255) )

                # Since we don't use the rendered images for this dataset,
                # randomly mask out parts of the object to account for occlusion.
                #num_erase = np.random.randint(0, 6)
                #for eraser in range(num_erase):
                #    # Erase somewhere near the gt bbox
                #    xywh = np.array(obj_info["bbox"], dtype=np.int)
                #    xywh += np.random.randint(0, 100, size=(4,))
                #    area = xywh[2] * xywh[3]
                #    target_area = random.uniform(0.02, 0.4) * area
                #    aspect_ratio = random.uniform(0.3, 1/0.3)

                #    h = int(round(math.sqrt(target_area * aspect_ratio)))
                #    w = int(round(math.sqrt(target_area / aspect_ratio)))
                #    x, y = xywh[:2]
                #    img0_bg_mask[y:y+h, x:x+w] = True

            else:
                # Use this hack so we don't have to waste time loading all the masks
                # For images with multiple objects, there are a lot of masks to load
                img0_bg_mask = depth0 == 0

            bg_image_path = self.bg_image_files[np.random.randint(len(self.bg_image_files))]
            bg0 = cv2.imread(bg_image_path)
            assert bg0 is not None, f"Empty image {bg_image_path}"
            assert bg0.size > 0, f"Empty image {bg_image_path}"
            bg0 = cv2.resize(bg0, img0.shape[:2][::-1])
            #cv2.imshow('bg', np.concatenate((img0, 
            #        cv2.cvtColor(255*img0_bg_mask.astype(np.uint8),cv2.COLOR_GRAY2BGR)), axis=1))
            #cv2.waitKey(5000)
            img0[img0_bg_mask] = bg0[img0_bg_mask]
        
        num_obj = len(obj_ids)
        bboxes = np.zeros((num_obj, 4), dtype=np.float32)
        
        # Make the bboxes in separate loop. We will then apply augmentations
        # before getting the keypoints, which can be calculated with the augmented 
        # K matrix.
        for i, obj_id in enumerate(obj_ids):
            obj_info = self.data[scene_id][view_id]["objects"][obj_id]
            xywh = np.array(obj_info["bbox"], dtype=np.float32)
            if '+noise' in self.det_type:
                xywh += np.random.normal(scale=20, size=(4,)).astype(np.float32)
            x, y, w, h = xywh
            w, h = max(10, w), max(10, h)
            bboxes[i] = np.array([x,y,x+w,y+h],dtype=np.float32)

        # Paste the paste images if any
        for img_paste, paste_mask in paste_imgs:
            # Perturb the image
            #perturbation = aug.NpScaleAndRotate()
            #img_paste = perturbation(img_paste)[0]

            # Choose a random place to paste it (near the bbox)
            x1,y1,x2,y2 = bboxes[np.random.randint(num_obj)].astype(np.int)
            px = min(max(0,np.random.randint(x1-img_paste.shape[1], x2)), 
                    img0.shape[1]-img_paste.shape[1])
            py = min(max(0,np.random.randint(y1-img_paste.shape[0], y2)),
                    img0.shape[0]-img_paste.shape[0])
            mask = img_paste
            ph, pw = img_paste.shape[0], img_paste.shape[1]
            img0[py:py+ph, px:px+pw][paste_mask] = img_paste[paste_mask]

        # Perform augmentations on image, depth, K, and bboxes
        # Since K has been fixed for any image transformation, the keypoints
        # will project correctly from 3D to the correct image plane.
        if not self.no_aug and random.random() < 0.8:
            for augmentation in self.img_aug:
                img0, depth0, bboxes, K = augmentation(img0, depth0, bboxes, K)

        data = {
            "img": aug.to_torch_uint8(img0).permute(2,0,1).to(torch.float32)/255,
            "K": torch.tensor(K.astype(np.float32)),
            "obj_ids": torch.tensor(obj_ids, dtype=torch.long),
            "bboxes": torch.tensor(bboxes),
        }

        # Note that this pose is just the GT pose without any care for symmetry
        poses = np.zeros((num_obj, 3, 4), dtype=np.float32)
        # ROI bboxes [x1,y1,x2,y2] that can be used for roi_align
        input_shape = IMAGE_SIZE
        #inputs = np.zeros((num_obj, 3+1+kp_config.num_kp(), input_shape[0], input_shape[1]),
        #        dtype=np.float32)
        priors = np.zeros((num_obj, kp_config.num_kp(), input_shape[0], input_shape[1]),
                dtype=np.float32)
        # Give the prior kp UVs as well in case we want it for another purpose
        prior_uvs = np.zeros((num_obj, kp_config.num_kp(), 2), dtype=np.float32)
        has_prior = np.zeros((num_obj), dtype=np.bool)
        K_kps = np.zeros((num_obj, 3, 3), dtype=np.float32)
        kp_uvs = np.zeros((num_obj, kp_config.num_kp(), 2), dtype=np.float32)
        kp_masks = np.zeros((num_obj, kp_config.num_kp()), dtype=np.bool)
        # 3D kp in CAD frames
        model_kps = np.zeros((num_obj, kp_config.num_kp(), 3), dtype=np.float32)
        kp_model_masks = np.zeros((num_obj, kp_config.num_kp()), dtype=np.bool)
        for i, obj_id in enumerate(obj_ids):
            obj_info = self.data[scene_id][view_id]["objects"][obj_id]
            kp_map = self.kp_map_per_object[obj_id-1]
            kp_list = self.kp_list_per_object[obj_id-1]
            T_OtoC = np.copy(obj_info["pose"])
            
            poses[i] = obj_info["pose"].astype(np.float32)
            
            # Give prior input 100*p_give_prior% of the time.
            # If testing, obviously the prior is idealized. 
            # Not to be used in actual evaluation, which will come from
            # actually projecting the 3D model from the estimated pose.
            # Otherwise, the user supplies prior.
            p_give_prior = 0.5
            #give_prior = "train" in self.split and \
            #give_prior = len(self.symmetries["discrete"][obj_id-1]) > 1 and \
            give_prior = np.random.rand() < p_give_prior
            has_prior[i] = give_prior
            
            #if "train" in self.split and not give_prior: 
            if not self.ignore_symmetry:
                T_OtoC, _ = self.pick_symmetry_transform(obj_id-1, T_OtoC, random=give_prior)

            # Project the keypoints from object frame onto cropped/resized image plane
            # p_FinC is [num_kp, 3, 1]
            p_FinC = utils.transform_pts(T_OtoC, self.gt_kp[obj_id-1]["kp_avg"])

            # Now from camera frame to image plane
            uvz_full = p_FinC @ K.T
            uv_depth_mm = uvz_full[:,2]
            uv_full = uvz_full[:,:2] / uv_depth_mm[:,None]
            
            # Whether or not to mask out occluded kp.
            if self.mask_occluded:
                # NN interp the depth map and see if the kp agree within a tolerance.
                # If they do, then the kp is most likely visible
                uv_idx = (.5 + uv_full).astype(np.int)
                uv_idx[:,0] = np.clip(uv_idx[:,0], 0, depth0.shape[1]-1)
                uv_idx[:,1] = np.clip(uv_idx[:,1], 0, depth0.shape[0]-1)
                uv_depth_mm_meas = depth0[uv_idx[:,1], uv_idx[:,0]]
                assert uv_depth_mm_meas.shape == uv_depth_mm.shape
                dtol = 10.0 # Difference tolerence (mm)
                depths_agree_mask = np.abs(uv_depth_mm_meas-uv_depth_mm) < dtol
            else:
                depths_agree_mask = np.ones((uv_full.shape[0]), dtype=np.bool)
                
            # Now account for crop and resize
            x, y, x2, y2 = bboxes[i]
            w, h = x2 - x, y2 - y
            duv = np.array([x,y], dtype=np.float64)
            kp_uv = uv_full - duv[None,:] # Pixels are shifted by bbox x,y

            # Make into normalized coordinates
            kp_uv[:,0] = 2*kp_uv[:,0]/w - 1
            kp_uv[:,1] = 1 - 2*kp_uv[:,1]/h
            # Fix the K matrix for the operations on UV
            K_i = utils.fix_K_for_bbox_ndc(K, bboxes[i])
            K_kps[i] = K_i.astype(np.float32)

            # Mask out out-of-bounds kp whether mask_occluded or not
            uv_in_bounds_mask = np.all(np.logical_and(kp_uv >= -1, kp_uv <= 1), axis=1)

            # Now, using the kp_config, fill in a tensor for all keypoints for all objects
            # (some shared between categories) and set a mask (just a vector here) for
            # which keypoints are valid
            kp_full = np.zeros((kp_config.num_kp(), 2), dtype=np.float32)
            # Mask that tells us which KP predicted are valid (i.e., visible or easily guessed)
            kp_mask = np.zeros((kp_config.num_kp()), dtype=np.bool)
            # KP in CAD frame. Only needed for eval
            kp_model = np.zeros((kp_config.num_kp(), 3), dtype=np.float32)
            # Mask that tells us which KP from the CAD model are valid
            kp_mask_model = np.zeros((kp_config.num_kp()), dtype=np.bool)

            # This will tell us the mapping from kp_uv indices to kp_full
            # TODO just make kp_mask_model at startup and use the vectorized mask ops
            for kp_uv_idx, kp_name_str in enumerate(kp_list):
                kp_full_idx = kp_map[kp_name_str]
                kp_full[kp_full_idx] = kp_uv[kp_uv_idx].astype(np.float32)
                kp_model[kp_full_idx] = \
                        self.gt_kp[obj_id-1]["kp_avg"][kp_uv_idx].astype(np.float32)
                kp_mask_model[kp_full_idx] = True
                # Predict the "always_predict" kp even if it's behind the object, but not
                # if out-of-bounds bc soft-argmax cannot handle that case.
                #kp_always_pred = kp_config.kp_list[kp_full_idx] in kp_config.kp_always_pred_list
                #kp_mask[kp_full_idx] = (kp_always_pred or depths_agree_mask[kp_uv_idx]) \
                kp_mask[kp_full_idx] = depths_agree_mask[kp_uv_idx] and uv_in_bounds_mask[kp_uv_idx]
                
            kp_uvs[i] = kp_full.astype(np.float32)
            kp_masks[i] = kp_mask
            model_kps[i] = kp_model
            kp_model_masks[i] = kp_mask_model

            if give_prior:
                # Get the Gaussian blurred prior keypoint prediction.
                # Make the prior noisy by perturbing the points with some common
                # rigid transformation, since the points will come from projecting 
                # all the CAD model points. Also, just give all the keypoints as prior
                # without masking out since why not (out-of-bounds kp will not appear clearly).
                # Do this no matter what no_aug and det_type are.
                dT = np.eye(4)
                dT[:3,:3] = utils.euler2R(np.random.normal(scale=5, size=(3,)))
                dT[:3,3] = np.array([np.random.normal(scale=s) for s in [5,5,10]])
                T_OtoC_4x4 = np.eye(4)
                T_OtoC_4x4[:3,:] = T_OtoC[:3,:]
                #p_FinC_noisy = utils.transform_pts(T_OtoC @ dT, kp_model)
                p_FinC_noisy = utils.transform_pts(dT @ T_OtoC_4x4, kp_model)
                kp_full_noisy = p_FinC_noisy @ K_i.T
                kp_full_noisy = kp_full_noisy[:,:2] / kp_full_noisy[:,2:3]
                priors[i] = utils.make_prior_kp_input(kp_full_noisy, kp_mask_model, input_shape)
                prior_uvs[i] = kp_full_noisy

        #data["segmask"] = segmask
        data["poses"] = torch.tensor(poses)
        data["priors"] = torch.tensor(priors)
        data["prior_uvs"] = torch.tensor(prior_uvs)
        data["has_prior"] = torch.tensor(has_prior)
        #data["inputs"] = inputs
        data["K_kps"] = torch.tensor(K_kps)
        data["kp_uvs"] = torch.tensor(kp_uvs)
        data["kp_masks"] = torch.tensor(kp_masks)
        data["model_kps"] = torch.tensor(model_kps)
        data["kp_model_masks"] = torch.tensor(kp_model_masks)
        return data

