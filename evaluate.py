#!/usr/bin/env python3

import os
import cv2
import json
import torch
import shutil
import subprocess
import numpy as np
from time import time

from lib.models import hg
from lib.datasets import bop
import lib.utils.utils as utils
from lib.labeling import kp_config
from lib.object_slam import ObjectSLAM
from lib.utils.eval_meter import EvalMeter
from lib.utils.training_utils import DataParallelWrapper
from thirdparty.bop_toolkit.bop_toolkit_lib.inout import load_ply
from lib.utils.mesh_database import load_mesh_db, load_mesh_db_DEBUG

YCBV_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

TLESS_CLASSES = {}
for i in range(30):
    TLESS_CLASSES[i+1] = str(i+1)

class Evaluator:
    def __init__(self, dataset, data_root, chkpt_path, 
            nviews=1, no_network_cov=False, detection_type='saved', 
            debug_gt_kp=False, gt_cam_pose=False, no_prior_det=False,
            show_viz=False, viz_cov=False, no_viz=False, 
            do_viz_extra=False, debug_saved_only=False, give_all_prior=False):
        
        self.model_path = os.path.dirname(chkpt_path)
        kp_var_thresh = 0.2
        bbox_thresh = 0.9
        bbox_inflate = 0.0
        opt_init_with_outliers = False
        if dataset == "ycbv":
            models = "models_bop-compat_eval"
            split = "test"
            self.do_add = True
            manual_kp_std = 0.01 # Close avg network STD for this dataset
        elif dataset == "tless":
            models = "models_eval"
            split = "test_primesense"
            self.do_add = False
            kp_var_thresh = 0.5
            bbox_thresh = 1.0
            manual_kp_std = 0.1 # Manually tuned
            opt_init_with_outliers = True
        else:
            assert False
        self.dataset = bop.BopDataset(data_root, split, bop_dset=dataset, ignore_symmetry=True)
        self.mesh_db = load_mesh_db(os.path.join(data_root, models))

        self.debug_saved_only = debug_saved_only
        if not self.debug_saved_only:
            self.object_slam = ObjectSLAM(chkpt_path, self.mesh_db, no_network_cov=no_network_cov,
                    no_prior_det=no_prior_det, debug_gt_kp=debug_gt_kp, sfm_mode=nviews>0,
                    single_view_mode=nviews==1, viz_cov=viz_cov, do_viz_extra=do_viz_extra,
                    kp_var_thresh=kp_var_thresh, bbox_thresh=bbox_thresh, 
                    bbox_inflate=bbox_inflate, manual_kp_std=manual_kp_std,
                    opt_init_with_outliers=opt_init_with_outliers, give_all_prior=give_all_prior)
        self.nviews = nviews
        if self.nviews == 1:
            print("NOTE: Running in single-view mode (nviews=1)")
        elif self.nviews < 0:
            print("NOTE: Running in SLAM mode (nviews=-1)")
        else:
            print(f"NOTE: Running in SfM mode (nviews={nviews})")
        self.detection_type = detection_type
        self.debug_gt_kp = debug_gt_kp
        self.gt_cam_pose = gt_cam_pose
        self.no_viz = no_viz
        self.show_viz = not no_viz and show_viz
        if self.show_viz:
            cv2.namedWindow("ObjectSLAM")

        self.saved_detections = None
        if detection_type == 'saved':
            # Load saved detections (PoseCNN for YCBV) and map by scene_id, view_id, obj_id
            if dataset == "ycbv":
                self.saved_detections = utils.load_posecnn_results(self.dataset.bop_root)
            elif dataset == "tless":
                self.saved_detections = utils.load_pix2pose_results(self.dataset.bop_root)
            else:
                assert False
            self.saved_detections_map = {} 
            for i in range(len(self.saved_detections["view_ids"])):
                scene_id = self.saved_detections["scene_ids"][i]
                view_id = self.saved_detections["view_ids"][i]
                obj_id = self.saved_detections["obj_ids"][i]
                if scene_id not in self.saved_detections_map.keys():
                    self.saved_detections_map[scene_id] = {}
                if view_id not in self.saved_detections_map[scene_id].keys():
                    self.saved_detections_map[scene_id][view_id] = {}
                assert obj_id not in self.saved_detections_map[scene_id][view_id].keys(), \
                        "Found duplicate object in saved detections"
                if self.dataset.targets is None \
                        or obj_id in self.dataset.targets.get(scene_id,{}).get(view_id,[]):
                    self.saved_detections_map[scene_id][view_id][obj_id] = i

    def run(self):
        t0 = time()
        try:
            self.__run()
        except:
            print()

            import traceback
            # Print stack trace
            traceback.print_exc()
        print(f"Eval took {time()-t0:.3f} sec")

    def __run(self):
        if self.saved_detections is not None:
            # Eval PoseCNN or other saved detections to validate eval code
            self.saved_det_meter = EvalMeter(self.mesh_db)
        
        num_cam_poses_found = 0
        num = 0

        if not self.debug_saved_only:
            self.meter = EvalMeter(self.mesh_db)
            csv_lines = []

            method = f"pkpnet-epoch={self.object_slam.model_epoch}" \
                   + f"-nviews={self.nviews}-det={self.detection_type}"
            if self.debug_gt_kp:
                method += "-GT-KP"
            if self.gt_cam_pose:
                method += "-GT-CAM-POSE"
            if self.object_slam.give_all_prior:
                method += "-ALL-PRIOR"
            if self.object_slam.no_network_cov:
                method += "-NO-COV"
            if self.object_slam.no_prior_det:
                method += "-NO-PRIOR-DET"
            method += f"_{self.dataset.bop_dset}-{self.dataset.split}"
            outdir = os.path.join(self.model_path, method)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            print(f"Writing eval results to {outdir}")
            viz_path = os.path.join(outdir, "viz_images")
            if not os.path.exists(viz_path):
                os.makedirs(viz_path)
            print(f"Writing eval vizualizations to {viz_path}")

        scene_ids = self.dataset.scene_ids()
        #for i, scene_id in enumerate(scene_ids[5:6]): # YCBV scene 53 with bowl
        for i, scene_id in enumerate(scene_ids):
            #break # DEBUG if you have bop csv already written and just want to skip to that
            view_ids = self.dataset.view_ids(scene_id)
            if not self.debug_saved_only and self.nviews < 0:
                self.object_slam.reset()

            # Store the results for whole scene then eval all at once at the end so that
            # we get the most optimized result for nviews < 0 (i.e. SLAM mode)
            scene_results = []

            for j, view_id in enumerate(view_ids):
                print_ln = f"Running scene [{i+1}/{len(scene_ids)}] view [{j+1}/{len(view_ids)}]"
                gt_obj_ids = self.dataset.obj_ids(scene_id, view_id)
                if self.debug_saved_only:
                    print(print_ln + ' '*(80-len(print_ln)), end='\r', flush=True)
                else:
                    print("==============================================================")
                    print(print_ln)
                    views_to_proc = [view_id]
                    if self.nviews > 1:
                        # Sample nviews-1 other views (excluding view_id)
                        views_to_proc += np.random.choice(view_ids[:j] + view_ids[j+1:],
                                size=self.nviews-1, replace=False).tolist()
                    results = self.__run_slam(scene_id, views_to_proc)
                    if len(results) == 0:
                        continue
                    
                    if not self.no_viz:
                        # Make vizualization (concating viz from all views if doing multiview)
                        viz_tot = None
                        view_ids_viz = self.object_slam.view_ids if self.nviews > 0 \
                                else [self.object_slam.view_ids[-1]]
                        for view_id_ba in view_ids_viz:
                            if viz_tot is None:
                                viz_tot = results[view_id_ba]["viz"]
                            else:
                                # Concatenated VERTICALLY
                                viz_tot = np.concatenate((viz_tot, results[view_id_ba]["viz"]), 
                                                          axis=0)
                        if self.show_viz:
                            cv2.imshow("ObjectSLAM", viz_tot)
                            cv2.waitKey(1)
                        fn = os.path.join(viz_path, f"scene_{scene_id}_{j:06d}.png")
                        cv2.imwrite(fn, viz_tot)
                        print("Wrote viz to", fn)

                        # Extra visualizations for figures
                        if self.object_slam.do_viz_extra:
                            viz_extra_dir = os.path.join(viz_path, f"scene_{scene_id}_{j:06d}")
                            if not os.path.exists(viz_extra_dir):
                                os.makedirs(viz_extra_dir)
                            for k, v in results[view_ids_viz[0]]["viz_extra"].items():
                                if v.size > 0:
                                    cv2.imwrite(os.path.join(viz_extra_dir, f"{k}.png"), v)
                                    print("Wrote viz to", os.path.join(viz_extra_dir, f"{k}.png"))
                        

                    # Wait til final opt to get pose for SLAM mode
                    pred_poses = results[view_id]["poses"] if self.nviews > 0 else None
                    scene_results.append( (view_id, pred_poses, gt_obj_ids) )

                # Update the saved_detection eval meter (other is saved for later
                # in case of SLAM mode).
                # Just grab results for view_id, which is the reference view here.
                if self.do_add and self.saved_detections is not None:
                    for gt_obj_id in gt_obj_ids:
                        if gt_obj_id in self.saved_detections_map[scene_id][view_id].keys():
                            sdet_idx = self.saved_detections_map[scene_id][view_id][gt_obj_id]
                            self.saved_det_meter.update([gt_obj_id], 
                                    self.saved_detections["poses"][sdet_idx][None,...],
                                    self.dataset.get_obj_pose(
                                            scene_id, view_id, gt_obj_id)[None,...])
                        else:
                            self.saved_det_meter.update_no_det([gt_obj_id])
                

            if not self.debug_saved_only:
                # Eval the results for the whole scene at once here
                print("Updating eval data...")
                # If in SLAM mode collect the results now
                if self.nviews < 0:
                    final_results = self.object_slam.collect_results(no_viz=True, final=True)
                for view_id, pred_poses, gt_obj_ids in scene_results:
                    num += 1
                    if self.nviews < 0:
                        if view_id not in final_results.keys():
                            if self.do_add:
                                for obj_id in gt_obj_ids:
                                    self.meter.update_no_det([obj_id])
                            continue
                        else:
                            num_cam_poses_found += 1
                        pred_poses = final_results[view_id]["poses"]

                    for obj_id in gt_obj_ids:
                        if obj_id in pred_poses.keys() and pred_poses[obj_id]["T_OtoC"] is not None:
                            result = pred_poses[obj_id]
                            gt_pose = self.dataset.get_obj_pose(scene_id, view_id, obj_id)
                            if self.do_add:
                                self.meter.update([obj_id], result["T_OtoC"][None,...], 
                                        gt_pose[None,...])
                            R, t = result["T_OtoC"][:3,:3], result["T_OtoC"][:3,3]
                            def arr2str(x):
                                return " ".join(str(elt) for elt in x.reshape(-1).tolist()) 
                            line = f"{scene_id},{view_id},{obj_id},{result['score']},"
                            line += f"{arr2str(R)},{arr2str(t)},-1"
                            if self.dataset.is_target(scene_id, view_id, obj_id):
                                csv_lines.append(line + "\n")
                        else:
                            print(f"NOTE: Could not obtain object pose for object {obj_id}")
                            self.meter.update_no_det([obj_id])
        
        if self.do_add:
            if self.dataset.bop_dset == 'ycbv':
                gt_obj_map = YCBV_CLASSES
            elif self.dataset.bop_dset == 'tless':
                gt_obj_map = TLESS_CLASSES
            else:
                assert False
            if self.saved_detections is not None:
                print("\nSaved PoseCNN result:")
                self.saved_det_meter.pprint_objs(gt_obj_map)
                #self.saved_det_meter.pprint()

        if not self.debug_saved_only:
            if self.do_add:
                print(f"\n{method} result:")
                self.meter.pprint_objs(gt_obj_map)
            summ_path = os.path.join(outdir, "summary.txt")
            with open(summ_path, 'w') as f:
                print(f"Writing this result to {summ_path}")
                if self.do_add:
                    f.write(self.meter.pprint_objs_str(gt_obj_map))
                # Write these items whether doing ADD* or not. 
                if num > 0:
                    ss = [f"NOTE: {100*num_cam_poses_found/num:.1f}% of camera poses found!",
                           self.object_slam.get_tracking_strtime(),
                           self.object_slam.get_global_opt_strtime(),
                          f"Average keypoint stdev: {self.object_slam.avg_std_meter.average()}"]
                    for s in ss:
                        print(s)
                        f.write("\n" + s + "\n")

        if not self.debug_saved_only:
            csv_path = os.path.join(outdir,  method + ".csv")
            with open(csv_path, "w") as f:
                f.writelines(csv_lines)
            print(f"\n\nCSV (BOP format) results written to {csv_path}\n")
            if self.dataset.bop_dset == "tless":
                print("Running BOP eval script...")
                myenv = os.environ.copy()
                myenv['PYTHONPATH'] = os.path.realpath('thirdparty/bop_toolkit/')
                # Avoid realpath here in case it's symlinked. Found that that can fail.
                myenv['BOP_PATH'] = os.path.join(os.getcwd(), 'data/bop_datasets/')
                assert os.path.exists(myenv['PYTHONPATH'])
                subprocess.call(['python', 'scripts/eval_siso.py',
                                 '--renderer_type', 'python',
                                 '--result_filename', os.path.realpath(csv_path),
                                 '--results_path', '',
                                 '--eval_path', os.path.realpath(outdir),
                                 '--targets_filename', self.dataset.targets_filename],
                                env=myenv, cwd='thirdparty/bop_toolkit/')

    def __run_slam(self, scene_id, views_to_proc):
        """
        views_to_proc: list of int views to process
        """
        # Delete SLAM data and treat these views as their own problem unless processing
        # the whole sequence, in which case we keep the prev data and assume there's
        # just one view here.
        if self.nviews > 0:
            self.object_slam.reset()
        else:
            assert len(views_to_proc) == 1
        for k, view_id_k in enumerate(views_to_proc):
            # Get object info for the detections
            obj_ids = []
            obj_ids_gt = self.dataset.obj_ids(scene_id, view_id_k)
            if 'gt' in self.detection_type:
                obj_ids = obj_ids_gt
            else:
                # Only take the detections that are of objects in this scene.
                # If they are not in the scene, then we can't eval them anyways.
                obj_keys = self.saved_detections_map.get(scene_id,{}).get(view_id_k,{}).keys()
                for obj_id in obj_keys:
                    if obj_id in obj_ids_gt:
                        obj_ids.append(obj_id)
                assert len(obj_ids) == len(set(obj_ids)), "Duplicates in detections?"
                if len(obj_ids) == 0:
                    print(f"\nWARNING no detections for scene {scene_id} view {view_id_k}")
                    print("Detection IDs:", list(obj_keys))
                    print("GT IDs:", obj_ids_gt)
                    continue
            
            sample = self.dataset.get_raw(scene_id, view_id_k, obj_ids)
            if 'gt' in self.detection_type:
                bboxes = sample["bboxes"].numpy()
            else:
                det_idx = self.saved_detections_map[scene_id][view_id_k][obj_ids[0]]
                bboxes = [self.saved_detections["bboxes"][self.saved_detections_map[ \
                        scene_id][view_id_k][obj_id]] for obj_id in obj_ids]
        
            # IF debugging with GT cam pose, grab it here.
            cam_pose = None
            if self.gt_cam_pose:
                # Make the poses relative to first view
                cam_pose = self.dataset.get_cam_pose(scene_id, view_id_k) \
                         @ utils.invert_SE3(self.dataset.get_cam_pose(     \
                           scene_id, -1 if self.nviews<0 else views_to_proc[0]))

            # Build the keypoint network batch input based on detected bboxes
            # TODO roi_align would be so much easier.
            img_np = (255 * sample["img"].numpy().transpose((1,2,0))).astype(np.uint8)
            K_np = sample["K"].numpy()   
            self.object_slam.process_view(view_id_k, img_np, K_np,
                    np.array(obj_ids,dtype=np.int), np.array(bboxes), sample["model_kps"].numpy(),
                    sample["kp_model_masks"].numpy(), sample["kp_masks"].numpy(), 
                    uv_gt = sample["kp_uvs"].numpy() if self.debug_gt_kp else None,
                    cam_pose = cam_pose)
            
        return self.object_slam.collect_results(last_only=self.nviews<0, no_viz=self.no_viz)
         
    
if __name__ == '__main__':
    from lib.args import get_args
    args = get_args('eval')
    if args.debug_gt_kp:
        args.detection_type = "gt"
    print("======= Eval Args ================")
    for attr in dir(args):
        if not attr.startswith('_'):
            print(f"{attr}: {getattr(args, attr)}")
    print("=============================")
    np.random.seed(666)
    Evaluator(args.dataset, args.data_root, args.checkpoint_path, nviews=args.nviews, 
            no_network_cov=args.no_network_cov, detection_type=args.detection_type, 
            debug_gt_kp=args.debug_gt_kp, gt_cam_pose=args.gt_cam_pose,
            no_prior_det=args.no_prior_det, show_viz=args.show_viz, viz_cov=args.viz_cov,
            no_viz=args.no_viz, debug_saved_only=args.debug_saved_only, 
            do_viz_extra=args.do_viz_extra, give_all_prior=args.give_all_prior).run()
    os.system('notify-send SUO-SLAM "Eval completed"')
