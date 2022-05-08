import os
import cv2
import torch
import numpy as np
from time import time
from collections import defaultdict

# Thirdparty ##################
import g2o 
import lambdatwist
################################

from .utils import utils
from .labeling import kp_config
from .models.pkpnet import PkpNet
from .utils.eval_meter import AverageMeter

from thirdparty.bop_toolkit.bop_toolkit_lib.renderer_py import RendererPython

# Return PnP pose only if enough points otherwise None.
# Returns an estimated transformation that transforms points in same
# frame as points_3d to the camera frame with corresponding image plane
# points_2d are defined in as well as a mask of inliers if ransac is set.
# If no ransac, inliers is just all True.
def pnp(points_3d, points_2d, camera_matrix):
    assert points_3d.shape[0] == points_2d.shape[0], \
            'points 3D and points 2D must have same number of rows'
    assert camera_matrix.shape == (3,3), "Camera matrix must be of shape (3,3)"
   
    num_pts = points_3d.shape[0]
    if num_pts < 4: # LambdaTwist PnP RANSAC need >=4 pts.
        return None 
        
    # Have to normalize the image coords
    KinvT = np.linalg.inv(camera_matrix).T
    points_2d_norm = points_2d @ KinvT[:2,:2] + KinvT[2:3,:2]
    res = lambdatwist.pnp(points_3d, points_2d_norm)
    if np.allclose(res, np.eye(4)):
        return None
    else:
        return res[:3,:], np.ones((num_pts), dtype=np.bool)

# This class will handle the logic of running the keypoint network
# with or without the prior detection, calculating the prior detection,
# and optimizing the results with a BA. All user does is instantiate,
# feed image and bbox. Note that each image can have a different K matrix. 
# We offer a single_view_mode for just performing covariance-driven PnP like PVNet
# and SfM mode for dealing with unordered sets of images in non-realtime, although
# this functionality is not as good as the baseline SLAM with all views since
# we don't do anything special for ordered images vs. non-ordered.
class ObjectSLAM:
    def __init__(self, chkpt_path, mesh_db, no_network_cov=False, 
            no_prior_det=False, pred_res=(256,256), debug_gt_kp=False,
            sfm_mode=False, single_view_mode=False, viz_cov=False, 
            do_viz_extra=False, global_opt_every=10, kp_var_thresh=0.2,
            bbox_thresh=0.9, bbox_inflate=0.0, manual_kp_std=0.005,
            opt_init_with_outliers=False, give_all_prior=False):
        """
        \param chkpt_path: string of path to load keypoint model from
        """
        
        self.mesh_db = mesh_db
        # Ignore network covariance if debugging with GT keypoints
        self.no_network_cov = no_network_cov or debug_gt_kp
        # Still make prior detection if debugging with GT keypoints so we
        # can debug that too.
        self.no_prior_det = no_prior_det
        self.pred_res = list(pred_res)
        self.debug_gt_kp = debug_gt_kp
        self.sfm_mode = sfm_mode 
        self.single_view_mode = single_view_mode
        self.slam_mode = not (sfm_mode or single_view_mode) 
        self.viz_cov = viz_cov
        self.do_viz_extra = do_viz_extra
        self.global_opt_every = global_opt_every
        self.kp_var_thresh = kp_var_thresh
        self.bbox_thresh = bbox_thresh
        self.bbox_inflate = bbox_inflate
        self.manual_kp_std = manual_kp_std
        self.opt_init_with_outliers = opt_init_with_outliers
        self.give_all_prior = give_all_prior

        self.reset()
        
        self.model = None
        self.model_epoch = -1
        if not debug_gt_kp:
            print(f"Loading model from {chkpt_path}")
            assert os.path.isfile(chkpt_path), \
                    "=> no checkpoint found at '{}'".format(chkpt_path)
            print("=> loading checkpoint '{}'".format(chkpt_path))
            checkpoint = torch.load(chkpt_path)
            self.model = PkpNet(calc_cov = not self.no_network_cov)
            self.model.load_state_dict(checkpoint['model'])
            assert self.model.calc_cov != self.no_network_cov 
            model_args = checkpoint['args']
            self.model_epoch = checkpoint['epoch']
            print("======= Model's Training Args ================")
            for attr in dir(model_args):
                if not attr.startswith('_'):
                    print(f"{attr}: {getattr(model_args, attr)}")
            print("=============================")
            if torch.cuda.is_available():
                print(f"Found CUDA")
                self.model = self.model.cuda()
            else:
                print("WARNING: No CUDA found.")
            self.model.eval()

        # Set the timers outside of the reset function in case we 
        # run multiple sequences.
        # Check the average std of the predicted keypoints
        self.avg_std_meter = AverageMeter()

        # We count the tracking time as the time for the network prediction, PnP,
        # and pose camera pose estimation. The graph opt and outlier rejection 
        # can be done in a seperate thread.
        self.track_time_meter = AverageMeter()

        self.opt_time_meter = AverageMeter()
        self.all_time_num_views = 0

        self.renderer = None

    def reset(self):
        '''
        Reset to initial state and delete old fed-in data if there is any.
        '''

        # Map view_id to dict of detection info 
        # (object-specific masks, bboxes, and keypoints mapped by object instance ID)
        self.detections = {}

        # Map view_id to current estimated camera pose for that view.
        # Pose is T_GtoC, which transforms points in world frame into camera frame.
        self.cam_poses = {}
        self.view_ids = [] # Order received of views 

        # K matrix for each camera pose. May be different.
        self.cam_K = {}
        self.images = {}

        # Map of object instance ID to object pose expressed in the first camera frame.
        # Pose is T_OtoG, which transforms points in object frame into world frame 
        # (first camera frame).
        self.obj_poses = {}

        # Info needed for deciding whether or not to re-initialize object.
        self.obj_num_dets = defaultdict(int)
        self.obj_num_det_kps = defaultdict(int)
        self.remove_penalty = defaultdict(int)
        
        self.needs_opt = False

    def get_global_opt_strtime(self, t0=np.nan, t1=np.nan):
        return f"TIMING: Global opt time: {1000*(t1-t0)} ms "     \
             + f"({1000*self.opt_time_meter.average()} avg) "     \
             + f"({1/self.opt_time_meter.average()} Hz)"

    def get_tracking_strtime(self, tt0=np.nan, tt1=np.nan):
        ttavg = self.track_time_meter.average()
        return f"TIMING: Tracking time: {1000*(tt1-tt0):.3f} ms " \
             + f"({1000*ttavg:.3f} avg) "                         \
             + f"({'inf' if ttavg<1e-12 else 1/ttavg} Hz)"
    
    def print_global_opt_time(self, t0=np.nan, t1=np.nan):
        print(self.get_global_opt_strtime(t0, t1))

    def print_tracking_time(self, tt0=np.nan, tt1=np.nan):
        print(self.get_tracking_strtime(tt0, tt1))

    '''
    Call this to get object poses in each camera frame for evaluation.
    '''
    def collect_results(self, last_only=False, no_viz=False, final=False):
        if self.slam_mode and self.needs_opt and final:
            print("Performing FINAL global optimization")
            t0 = time()
            self.optimize()
            t1 = time()
            self.opt_time_meter.update(t1-t0)
            self.print_global_opt_time(t0, t1)

        results = {}
        assert len(self.view_ids) == len(self.cam_poses)
        view_ids = [self.view_ids[-1]] if last_only else self.view_ids
        for view_id in view_ids:
            T_GtoC = self.cam_poses[view_id]
            results[view_id] = {
                "poses": {},
            }
            poses = {}
            detection = self.detections[view_id]
            if T_GtoC.shape[0] == 3:
                T_GtoC = np.concatenate((T_GtoC,np.eye(4)[3:4,:]), axis=0)
                
            # Combine estimated objects with objects detected in this frame.
            obj_ids = set(list(self.obj_poses.keys()) + list(detection.keys()))

            kp_cov = None
            if not no_viz:
                # Populate with keypoints detected in this view
                kp_pred = np.zeros((len(obj_ids), kp_config.num_kp(), 2), dtype=np.float32)
                if not self.no_network_cov:
                    kp_cov = np.zeros((len(obj_ids), kp_config.num_kp(), 2, 2), 
                            dtype=np.float32)
                kp_mask = np.zeros((len(obj_ids), kp_config.num_kp()), dtype=np.bool)
                bboxes = np.zeros((len(obj_ids), 5), dtype=np.int)
                # Remake the priors for the whole image now
                img = self.images[view_id]
                priors = np.zeros((kp_config.num_kp(),*img.shape[:2]), dtype=np.float32)

            for i, obj_id in enumerate(obj_ids):
                T_OtoC = None
                if obj_id in self.obj_poses.keys():
                    T_OtoG = self.obj_poses[obj_id]
                    if T_OtoG.shape[0] == 3:
                        T_OtoG = np.concatenate((T_OtoG,np.eye(4)[3:4,:]), axis=0)
                    T_OtoC = T_GtoC @ T_OtoG
                    poses[obj_id] = T_OtoC
                result = {
                    "T_OtoC": T_OtoC,
                    #"score": 0, 
                    "score": 1 + self.obj_num_inliers(obj_id), # TOTAL number of final inliers
                }
                if obj_id in detection.keys():
                    det_obj = detection[obj_id]
                    #result["score"] = np.count_nonzero(det_obj["inliers"])

                    if not no_viz:
                        kp_mask_i = det_obj["kp_mask"] # Which kp were detected
                        #inliers_i = det_obj["inliers"] # Which kp are final inliers
                        # Project kps into full image plane with this homography
                        H = (self.cam_K[view_id] @ np.linalg.inv(det_obj["K"])).T
                        kp_pred[i][kp_mask_i] = det_obj["uv_pred"] @ H[:2,:2] + H[2:3,:2]
                        kp_mask[i] = kp_mask_i
                        bboxes[i][0] = obj_id
                        bboxes[i][1:] = (det_obj["bbox"] + 0.5).astype(np.int)
                        if not self.no_network_cov:
                            assert "cov_pred" in det_obj.keys()

                            # Covariance propagate from the bbox image in ndc to full
                            # image in raw UV coordinates.
                            kp_cov[i][kp_mask_i] = H[:2,:2].T[None,None,...] \
                                    @ det_obj["cov_pred"] @ H[None,None,:2,:2]
                            #print(kp_cov[i][kp_mask_i])

                        if det_obj["prior_uv"] is not None:
                            prior_uv_full = det_obj["prior_uv"] @ H[:2,:2] + H[2:3,:2]

                            # Dont viz prior outside of bbox
                            x1,y1,x2,y2 = bboxes[i][1:]
                            priors[:,y1:y2,x1:x2] += utils.make_prior_kp_input(prior_uv_full,
                                    det_obj["model_kp_mask"], img.shape[:2], 
                                    ndc=False)[:,y1:y2,x1:x2]

                results[view_id]["poses"][obj_id] = result
            
            if not no_viz:
                # Get viz data
                priors = np.clip(priors,0,1)

                t0 = utils.device_time()
                # LEFT: bbox and prior, MIDDLE keypoints, RIGHT model overlay
                viz_img_left = utils.make_kp_viz(img, np.array([]), np.array([]), 
                        bbox_pred=bboxes, prior=priors)               
                viz_img_center = utils.make_kp_viz(img, kp_pred, kp_mask, 
                        cov=kp_cov if self.viz_cov else None, ndc=False)
                viz_img_right = utils.make_kp_viz(img, np.array([]), np.array([]), 
                        poses=poses, K=self.cam_K[view_id], mesh_db=self.mesh_db)
                viz_img = np.concatenate((viz_img_left, viz_img_center, viz_img_right), axis=1)
                results[view_id]["viz"] = viz_img
                t1 = utils.device_time()
                print(f"TIMING: Viz time = {1000 * (t1-t0)} ms")

                # Add some more visualizations for each object (for paper figs).
                if self.do_viz_extra:
                    results[view_id]["viz_extra"] = {}

                    # Image and bbox
                    results[view_id]["viz_extra"]["bbox_input"] = utils.make_kp_viz(
                            img, np.array([]), np.array([]), bbox_pred=bboxes)

                    for i, obj_id in enumerate(obj_ids):
                        x1,y1,x2,y2 = bboxes[i][1:]
                        img_i = img[y1:y2,x1:x2,:]
                        prior_i = priors[:,y1:y2,x1:x2]
                        kp_pred_i = kp_pred[i:i+1] - np.array([x1,y1])[None,None,:]    
                        T = np.eye(3, dtype=np.float32)
                        T[:2,2] = -np.array([x1,y1], dtype=np.float32)
                        K_bbox = T @ self.cam_K[view_id]
                        kp_cov_i = None
                        if kp_cov is not None:
                            kp_cov_i = kp_cov[i:i+1]

                        # Just image (and prior if nonzero)
                        results[view_id]["viz_extra"][f"viz_obj_{obj_id}_input"] = \
                                utils.make_kp_viz(img_i, np.array([]), np.array([]), prior=prior_i)
                        # Image with keypoints
                        results[view_id]["viz_extra"][f"viz_obj_{obj_id}_output"] = \
                                utils.make_kp_viz(img_i, kp_pred_i, kp_mask[i:i+1], 
                                        cov=kp_cov_i if self.viz_cov else None, ndc=False)
                        # Image overlayed with CAD model at estim. pose
                        if obj_id in poses.keys():
                            results[view_id]["viz_extra"][f"viz_obj_{obj_id}_overlay"] = \
                                    utils.make_kp_viz(img_i, np.array([]), np.array([]), 
                                    poses={obj_id: poses[obj_id]}, K=K_bbox, 
                                    mesh_db=self.mesh_db)
        return results

    def num_views_processed(self):
        return len(self.cam_poses)
    '''
    The main feed-in function.
    Args:
    view_id: int of the view_id. This should correspond to the order of the images in a video,
             but there can be gaps between view_ids (i.e., for keyframes).
    img: [h w 3] uint8 np.array
    K: [3 3] camera matrix for this image
    obj_ids: np.array of obj_ids
    bboxes: np.array of (x,y,w,h) detected bboxes for each obj_id.
    model_kps: np.array of 3D CAD keypoints for each object (each entry is [num_kp, 3] np.float32).
    kp_masks: np.array of masks for what keypoints to use for each object 
              (each entry is [num_kp] np.bool).
    '''
    # TODO model predicts own kp mask.
    def process_view(self, view_id, img, K, obj_ids, bboxes, model_kps, model_kps_masks,
            kp_masks, uv_gt=None, cam_pose=None):
        assert view_id not in self.cam_poses.keys(), \
                f"Repeat view_id {view_id} when already processed views {self.cam_poses.keys()}"
        
        tt0 = utils.device_time()
        self.cam_K[view_id] = K
        self.images[view_id] = img
        self.all_time_num_views += 1

        if not self.no_prior_det:
            # Split objects into symmetric and non-symmetric. Use non-symmetric objects to
            # determine the camera pose, then make a prior detection for the symmatric ones.
            # Treat symmetric objects as non-symmetic if this is its first detection (i.e. no
            # estimate of this object yet). Also ignore the continuous symmetries since
            # the network predicts keypoints that define the continuous axes of symmetry.
            is_sym = np.array([self.mesh_db[obj_id]["is_symmetric"] for obj_id in obj_ids])
        else:
            # Treat all objects as non-symmetric
            is_sym = np.zeros((len(obj_ids,)), dtype=np.bool)

        # Optional external cam pose
        if cam_pose is not None: 
            print("PROCESS_VIEW: Camera pose already provided")
            self.cam_poses[view_id] = cam_pose
            self.view_ids.append(view_id)
            is_sym = np.ones((len(obj_ids),), dtype=np.bool)

        # If give_all_prior, consider all as symmetric to accomplish this
        if self.give_all_prior:
            print("PROCESS_VIEW: Giving all objects prior detection")
            is_sym = np.ones((len(obj_ids),), dtype=np.bool)

        # Ignore the symmetry if single-view mode.
        if self.single_view_mode:
            is_sym = np.zeros((len(obj_ids),), dtype=np.bool)
        
        is_non_sym = ~is_sym
        n_sym = np.count_nonzero(is_sym)
        n_non_sym = np.count_nonzero(is_non_sym)
        print(f"NEW VIEW obj_ids: {obj_ids}, n_sym: {n_sym}, n_non_sym: {n_non_sym}")
        if cam_pose is None and not self.single_view_mode:
            #assert n_non_sym > 0 or cam_pose is not None, \
            #        "Need cam pose if no non-symmetric objects are provided"
            if len(self.view_ids) > 0 and n_non_sym == 0:
                # Pick random objects to be non-symmetric.
                # If we pick randomly every time, then ideally the objects will get the prior
                # every other time or so.
                #non_sym_inds = np.random.choice(len(is_non_sym), 
                #        size=len(is_non_sym)//2,replace=False)
                #is_non_sym[non_sym_inds] = True

                #is_sym = ~is_non_sym
                #n_sym = np.count_nonzero(is_sym)
                #n_non_sym = np.count_nonzero(is_non_sym)
                
                #print(f"Not enough non-symmetric objects. Assigning symmetry randomly. "
                #      f"NEW: n_sym: {n_sym}, n_non_sym: {n_non_sym}")
                print(f"Not enough non-symmetric objects. Attempting backup cam pose estimation. ")
                self.__backup_estimate_camera_pose(view_id, obj_ids, bboxes)

        self.needs_opt = True
        # Enlarge bboxes by a bit
        bboxes[:,[0,1]] *= 1. - self.bbox_inflate
        bboxes[:,[2,3]] *= 1. + self.bbox_inflate

        # Process non-symmetric object first
        if n_non_sym > 0:
            self.__process_objects(False, view_id, img, K, obj_ids[is_non_sym], bboxes[is_non_sym], 
                    model_kps[is_non_sym], model_kps_masks[is_non_sym], kp_masks[is_non_sym], 
                    uv_gt = uv_gt[is_non_sym] if uv_gt is not None else None)
        if view_id not in self.cam_poses.keys():
            if len(self.view_ids) == 0:
                self.view_ids.append(view_id)
                self.cam_poses[view_id] = np.eye(4)[:3,:]
            else:
                print("Non-symmetric camera pose estimation failed. "
                      "Attempting backup cam pose estimation.")
                self.__backup_estimate_camera_pose(view_id, obj_ids, bboxes)
        # Only run symmetric objects with prior det if camera pose could be recovered
        if n_sym > 0 and ((view_id in self.cam_poses.keys()) or self.no_prior_det):
            # Now try to initialize or track the symmetric ones.
            self.__process_objects(True, view_id, img, K, obj_ids[is_sym], bboxes[is_sym], 
                    model_kps[is_sym], model_kps_masks[is_sym], kp_masks[is_sym], 
                    uv_gt = uv_gt[is_sym] if uv_gt is not None else None)
        
        # Optimize the camera pose a bit and check for outlier measurements.
        if not self.single_view_mode:
            # Re-initialize objects which had bad initializations
            # and check for outliers with new pose.
            self.__maybe_reinit_objects(view_id, len(self.view_ids) if self.sfm_mode else 15)
            
            # Only optimize the cam pose if it was NOT provided
            print("Performing current-only camera optimization")
            self.optimize(curr_only=True)   

        tt1 = utils.device_time()
        # Warmup net before timing.
        if self.all_time_num_views > 5:
            self.track_time_meter.update(tt1-tt0)
        self.print_tracking_time(tt0, tt1)

        # Comment out "if cam_pose is not None" above, uncomment this, and
        # run with --gt_cam_pose to see this comparison between 
        # GT pose and estimated.
        # TODO actual trajectory comparison with RMSEs.
        #if cam_pose is not None and view_id in self.cam_poses.keys(): 
        #    print("\n\nEstimated: \n", self.cam_poses[view_id])
        #    print("GT: \n", cam_pose, "\n\n")
        
        # Refine the problem with full LM optimization 
        # For SfM mode (don't care about speed) and single-view mode (quick anyway with one view)
        # we do this for every new image, and for SLAM mode we do it periodically every
        # self.global_opt_every frames. Typically this would be in a nother thread, but
        # for just evaluation singe-thread is the only way to guarantee the same accuracy
        # every run.
        if self.sfm_mode or self.single_view_mode \
                or (len(self.view_ids) > 1 and len(self.view_ids) % self.global_opt_every == 0):
            print("Performing global optimization")
            t0 = time()
            self.optimize()
            t1 = time()
            self.opt_time_meter.update(t1-t0)
            self.print_global_opt_time(t0, t1)
            self.needs_opt = False

    # TODO Observations should be stored in each object not in this detections dict
    def obj_num_inliers(self, obj_id):
        num_inliers = 0
        for view_id, detection in self.detections.items():
            num_inliers += np.count_nonzero(detection.get(obj_id, {}).get("inliers",np.array([])))
        return num_inliers
    
    def remove_obj(self, obj_id):
        self.obj_poses.pop(obj_id)
        #self.obj_num_dets[obj_id] = 0
        #self.obj_num_det_kps[obj_id] = 0
        #self.remove_penalty[obj_id] = 0
        # Remove old (probably bad) detections
        #for view_id in self.detections.keys():
        #    if obj_id in self.detections[view_id].keys():
        #        self.detections[view_id].pop(obj_id)

    def __process_objects(self, is_sym, view_id, img, K, obj_ids, bboxes, 
            model_kps, model_kps_masks, kp_masks, uv_gt=None):
        
        if len(obj_ids) == 0:
            return
        assert len(obj_ids) == len(bboxes)
        assert len(obj_ids) == len(model_kps)
        assert len(obj_ids) == len(model_kps_masks)
        assert len(obj_ids) == len(kp_masks)

        prior_dets = None
        prior_det_uv = None

        # We will make the prior detection image if the camera pose is available
        # and the objects are symmetric.
        #if not self.no_prior_det and view_id in self.cam_poses.keys():
        if is_sym and (not self.no_prior_det) and (view_id in self.cam_poses.keys()):
            # Create the prior detection images
            prior_dets = {}
            prior_det_uv = {}
            T_GtoC = self.cam_poses[view_id]
            for k, obj_id in enumerate(obj_ids):
                # Project the 3D keypoints into the camera frame
                if obj_id in self.obj_poses.keys():
                    model_kp_mask = model_kps_masks[k]
                    kps_in_O = model_kps[k][model_kp_mask]
                    T_OtoG = self.obj_poses[obj_id]
                    if T_OtoG.shape[0] == 3:
                        T_OtoG = np.concatenate((T_OtoG, np.eye(4)[3:4,:]), axis=0)
                    T_OtoC = T_GtoC @ T_OtoG
                    kps_in_C = utils.transform_pts(T_OtoC, kps_in_O)
                    # Make the uvs into the normalized image plane of the bounding box.
                    K_bbox = utils.fix_K_for_bbox_ndc(K, bboxes[k])
                    uvd = kps_in_C @ K_bbox.T
                    if np.all(uvd[:,2] > 0):
                        #print(f"Adding prior obj_id={obj_id} (object {k+1}/{len(obj_ids)})")
                        uv = uvd[:,:2] / uvd[:,2:3]
                        # Place uv's in a bigger array containing all keypoints the net can predict
                        # Note that model_kps_mask is for all keypoints for this CAD model,
                        # and not just visible ones or anything
                        prior_uv_full = np.zeros((model_kp_mask.shape[0], 2), dtype=np.float32)
                        prior_uv_full[model_kp_mask] = uv # Fill in the projected keypoints
                        prior_det_uv[obj_id] = prior_uv_full
                        prior_dets[obj_id] = utils.make_prior_kp_input(
                                prior_uv_full, model_kp_mask, self.pred_res, ndc=True)
                    else:
                        print("WARNING: Skipping prior due to negative depth. "
                            + "Bad object or camera pose."
                            + f"\n\n==> [depth] = \n{uvd[:,2]}"
                            + f"\n==> T_OtoC = \n{T_OtoC}\n")

        kp_det = self.__run_kp_model(view_id, img, K, obj_ids, 
                bboxes, model_kps, model_kps_masks, kp_masks, uv_gt, prior_dets)
        if not self.no_network_cov:
            for det in kp_det:
                if det["cov_pred"] is not None:
                    pred_std = np.sqrt(det["cov_pred"][...,[0,1],[0,1]])
                    if pred_std.size > 0:
                        self.avg_std_meter.update(pred_std.mean(), pred_std.size)
            print(f"STD METER: Average predicted keypoint stdev: {self.avg_std_meter.average()}")

        detection = {}
        for k, obj_id in enumerate(obj_ids):
            detection[obj_id] = {
                "bbox": bboxes[k],
                "model_kp_mask": model_kps_masks[k],
                #"prior": prior_dets.get(obj_id) if prior_dets is not None else None,
                "prior_uv": prior_det_uv.get(obj_id) if prior_det_uv is not None else None,
            }
            for key, value in kp_det[k].items():
                detection[obj_id][key] = value
            if self.num_views_processed() == 0:
                # Set the initial object poses
                assert obj_id not in self.obj_poses.keys(), \
                        f"Object {obj_id} is in detections twice! " + \
                        "obj_id must be an instance label."
                # Onle add it if PnP was successful so that self.obj_poses are all valid.
                if detection[obj_id]["pose"] is not None:
                    T_OtoC = detection[obj_id]["pose"]
                    if view_id in self.cam_poses.keys():
                        # Cam pose was provided
                        T_GtoC = self.cam_poses[view_id]
                        T_OtoG = utils.invert_SE3(T_GtoC) @ T_OtoC
                    else:
                        # First cam pose is identity.
                        T_OtoG = T_OtoC
                    self.obj_poses[obj_id] = T_OtoG
        
        if view_id in self.detections.keys():
            for obj_id in obj_ids:
                assert obj_id not in self.detections[view_id].keys(), \
                        "Object has already been processed for this view"
                self.detections[view_id][obj_id] = detection[obj_id]
        else:
            self.detections[view_id] = detection
            
        # Determine the camera pose via the object points with one big PnP
        # if already initialized.
        # If camera pose is already present, then it was supplied by external odometry
        # or already calculated with a previous run of this function.
        if view_id not in self.cam_poses.keys():
            if self.num_views_processed() == 0:
                self.cam_poses[view_id] = np.eye(4)[:3,:]
            else:
                cam_pose = self.__estimate_camera_pose(view_id)
                if cam_pose is None: 
                    return
                self.cam_poses[view_id] = cam_pose
            self.view_ids.append(view_id)

        # Try to initialize objects that were not able to before.
        for k, obj_id in enumerate(obj_ids):
            # In this case, we were unable to initialize the object from past views.
            # So try it again. Note the object pose has to be transformed into
            # the world frame in this case.
            if obj_id not in self.obj_poses.keys() \
                    and obj_id in detection.keys() \
                    and detection[obj_id]["pose"] is not None:
                T_OtoC = detection[obj_id]["pose"]
                T_GtoC = self.cam_poses[view_id]
                T_OtoG = utils.invert_SE3(T_GtoC) @ T_OtoC
                self.obj_poses[obj_id] = T_OtoG
                # TODO initialize objects with continuous symmetries here 
                # by triangulating points/line for the kp on the axes               
        
    def __maybe_reinit_objects(self, view_id, check_n_views=15):
        '''
        Check if the PnP results for each object for the current detection has more inliers
        than the current 3d estimation. If so, reinitialize the object with the 
        pose from this PnP result and remove outliers from the problem. 
        This can combat a bad object initialization.
        If there are not at least 2 views, then do nothing.
        '''
        
        if self.num_views_processed() < 2 or view_id not in self.cam_poses.keys():
            return
        check_n_views = min(len(self.view_ids), check_n_views)

        t0 = time()
        curr_det = self.detections[view_id]
        # Match the object ids
        obj_ids = []
        for obj_id in self.obj_poses.keys():
            # We need the 3D estimate of the object, and a good PnP pose from the current frame.
            if curr_det.get(obj_id,{}).get("pose") is not None:
                obj_ids.append(obj_id)

        if len(obj_ids) == 0:
            return

        Ts_OtoG_estim = np.zeros((len(obj_ids),4,4), dtype=np.float32)
        for j in range(len(obj_ids)):
            Ts_OtoG_estim[j,:3,:] = self.obj_poses[obj_ids[j]][:3,:]
            Ts_OtoG_estim[j,3,3] = 1
        Ts_OtoC_pnp = np.stack([curr_det[obj_id]["pose"] if curr_det[obj_id]["pose"] \
                is not None else np.eye(4) for obj_id in obj_ids])

        # Compare Ts_OtoG_pnp to Ts_OtoG_estim
        Ts_OtoG_pnp = utils.invert_SE3(self.cam_poses[view_id])[None,:,:] @ Ts_OtoC_pnp
        views_to_check = [self.view_ids[-(i+1)] for i in range(check_n_views)]

        Ts_GtoCi = np.zeros((check_n_views,4,4), dtype=np.float32)
        for i in range(check_n_views):
            Ts_GtoCi[i,:3,:] = self.cam_poses[views_to_check[i]][:3,:]
            Ts_GtoCi[i,3,3] = 1
        
        Ts_OtoCi = {
                "pnp": Ts_GtoCi[:,None,:,:] @ Ts_OtoG_pnp[None,:,:,:], # check_n_views x num_obj
                "estim": Ts_GtoCi[:,None,:,:] @ Ts_OtoG_estim[None,:,:,:] # check_n_views x num_obj
        }
        num_reinit = 0
        for j in range(len(obj_ids)):
            obj_id = obj_ids[j]
            num_inliers = {
                    "estim": 0,
                    "pnp": 0,
            }
            for i in range(check_n_views):
                view_id_i = views_to_check[i]
                if obj_id in self.detections[view_id_i].keys():
                    if "pose" in curr_det[obj_id].keys():
                        for key in num_inliers.keys():
                            model_kp = self.detections[view_id_i][obj_id]["model_kp"]
                            p_FinC = utils.transform_pts(Ts_OtoCi[key][i,j], model_kp)
                        
                            # Make sure to use K from the bbox of this object in this view here.
                            uv_proj = p_FinC @ self.detections[view_id_i][obj_id]["K"].T
                            d_pos_mask = uv_proj[:,2] > 0
                            uv_proj = (uv_proj[:,:2] / uv_proj[:,2:3])[d_pos_mask]
                            if uv_proj.shape[0] > 0:
                                # Calculate chi2. Cov may be None if not using predicted covariance.
                                # Note that UVs are in normalized device coords, so manual
                                # covariance needs to be scaled properly for values in [-1,1]
                                uv = self.detections[view_id_i][obj_id]["uv_pred"][d_pos_mask]
                                cov = self.detections[view_id_i][obj_id]["cov_pred"]
                                res = uv - uv_proj
                                if cov is not None:
                                    cov = cov[d_pos_mask]
                                    # Ensure invertible
                                    cov[:,[0,1],[0,1]] = np.maximum(cov[:,[0,1],[0,1]], 1e-4)
                                    inf = np.linalg.inv(cov)
                                    assert not np.any(np.isnan(inf)), \
                                            f"NaN in information matrix!\nInf = {inf}\nCov = {cov}"
                                else:
                                    inf = np.zeros((res.shape[0],2,2), dtype=np.float32)
                                    sigma = self.manual_kp_std
                                    inf[:,[0,1],[0,1]] = 1 / sigma**2

                                chi2 = (res[:,None,:] @ inf @ res[:,:,None]).reshape(-1)
                                # inliers have a chi2 less than 5.991 for 2 DoF in 95% confidence.
                                chi2_inliers = chi2 <= 5.991
                                num_inliers[key] += np.count_nonzero(chi2_inliers)
            
            min_num_inliers = 3
            num_inliers["thresh"] = min_num_inliers
            print(f"RE-INIT checking object {obj_id} (num_inliers={num_inliers})")
            if "pnp" in num_inliers.keys() and num_inliers["pnp"] >= min_num_inliers \
                    and num_inliers["pnp"] > 3*num_inliers["estim"]:
                T_old = self.obj_poses[obj_id]
                self.obj_poses[obj_id] = Ts_OtoG_pnp[j]
                num_reinit += 1
                print(f"RE-INIT object {obj_id} needs to be reinitialized.")
                print("Old pose:")
                print(T_old)
                print("New pose:")
                print(self.obj_poses[obj_id])

        print(f"RE-INIT reinitialized {num_reinit} objects (t={1000 * (time()-t0):.3f} ms)\n")


    # If curr_only, this function only optimizes the current camera pose and objects,
    # similar to the tracking in ORB SLAM.
    # Otherwise, it does a global optimization.
    def optimize(self, curr_only=False):
        if len(self.view_ids) == 0:
            return
        optimizer = g2o.SparseOptimizer()
        if curr_only:
            solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        else:
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)
            
        # Count how many edges there are so we don't have unconnected components
        num_cam_edges = defaultdict(int) # Default to 0
        num_obj_edges = defaultdict(int)
        obj_ids = list(self.obj_poses.keys())
        view_id_curr = self.view_ids[-1] 
        if curr_only:
            view_id = view_id_curr
            if view_id not in self.cam_poses.keys():
                return
            detection = self.detections[view_id]
            if view_id in self.cam_poses.keys():
                for obj_id, det_data in detection.items():
                    if obj_id in obj_ids: # Else object couldn't be initialized
                        n_edges_det = np.count_nonzero(det_data["inliers"])
                        num_cam_edges[view_id] += n_edges_det
                        num_obj_edges[obj_id] += n_edges_det
                if num_cam_edges[view_id] < 3:
                    print("OPTIM CURRENT: Not enough measurements in current frame. Skipping")
                    return
            else:
                print("OPTIM CURRENT: Current view could not be estimated. Skipping")
                return
        else:
            for view_id, detection in self.detections.items():
                if view_id in self.cam_poses.keys():
                    for obj_id, det_data in detection.items():
                        if obj_id in obj_ids: # Else object couldn't be initialized
                            n_edges_det = np.count_nonzero(det_data["inliers"])
                            num_cam_edges[view_id] += n_edges_det
                            num_obj_edges[obj_id] += n_edges_det
        
        object_verts = {}
        for j, obj_id in enumerate(obj_ids):
            if num_obj_edges[obj_id] > 0:
                # Objects are fixed for curr_only. Just give the pose and don't add to graph.
                if curr_only:
                    object_verts[obj_id] = self.obj_poses[obj_id][:3,:]
                else:
                    # pose here means transform points from object frame to world
                    pose = g2o.SE3Quat(self.obj_poses[obj_id][:3,:3], self.obj_poses[obj_id][:3,3])
                    v_se3 = g2o.VertexSE3Expmap()
                    v_se3.set_id(j)
                    v_se3.set_estimate(pose)
                    optimizer.add_vertex(v_se3)
                    object_verts[obj_id] = v_se3
            
        camera_verts = {}
        cam_poses = {view_id_curr: self.cam_poses[view_id_curr]} if curr_only else self.cam_poses
        for i, view_id in enumerate(cam_poses.keys()):
            if num_cam_edges[view_id] > 0:
                # pose here means transform points from world coordinates to camera coordinates
                pose = g2o.SE3Quat(self.cam_poses[view_id][:3,:3], self.cam_poses[view_id][:3,3])
                v_se3 = g2o.VertexSE3Expmap()
                v_se3.set_id(i + (len(self.obj_poses) if not curr_only else 0))
                v_se3.set_estimate(pose)
                if curr_only:
                    v_se3.set_fixed(False)
                    #v_se3.set_marginalized(False)
                else:
                    # TODO need more fixed?
                    v_se3.set_fixed(i == 0)
                    # TODO Schur causes segfault, but it would make this faster
                    #v_se3.set_marginalized(True)
                optimizer.add_vertex(v_se3)
                camera_verts[view_id] = v_se3

        if len(camera_verts) == 0:
            print(f"WARNING: No cameras with graph edges. Cannot optimize. "
                  f"({self.num_views_processed()} views processed)")
            return
        if len(object_verts) == 0:
            print(f"WARNING: No objects with graph edges. Cannot optimize. "
                  f"({len(self.obj_poses)} objects estimated)")
            return
            
        detections = {view_id_curr: self.detections[view_id_curr]} if curr_only else self.detections
        edges = {}
        for view_id, detection in detections.items():
            edges[view_id] = {}
            for obj_id, det_data in detection.items():
                if view_id in camera_verts.keys() and obj_id in object_verts.keys():
                    K = det_data["K"]
                    assert np.allclose(K[[0,1,2,2,2],[1,0,0,1,2]], \
                           np.array([0,0,0,0,1],dtype=K.dtype)),    \
                          f"K matrix has off-diagonals!\n\n{K}"
                    cam_k = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                    # NOTE: We insert outlier meas into the problem now but
                    # with level==1 so its not optimized in the graph with level==0.
                    # This way, we can grab the chi2, and if its small enough, it can
                    # become an inlier again.

                    inlier_mask = det_data["inliers"]
                    uv = det_data["uv_pred"]
                    assert len(inlier_mask) == len(uv)
                    cov_uv = None
                    if det_data["cov_pred"] is not None:
                        cov_uv = det_data["cov_pred"]
                    model_pts = det_data["model_kp"]
                    edges_per_obj = []
                    for k in range(uv.shape[0]):
                        if curr_only:
                            # Unary edge, with fixed object pose
                            edge = g2o.EdgeSE3ProjectFromFixedObject(cam_k, model_pts[k],
                                    object_verts[obj_id]) # object_verts is just numpy array
                            edge.set_vertex(0, camera_verts[view_id])
                        else:
                            # Binary edge -- opt camera and object pose
                            edge = g2o.EdgeSE3ProjectFromObject(cam_k, model_pts[k])
                            edge.set_vertex(0, object_verts[obj_id])
                            edge.set_vertex(1, camera_verts[view_id])
                        edge.set_measurement(uv[k])
                        Inf = np.eye(2)
                        if cov_uv is not None:
                            Inf = np.linalg.inv(cov_uv[k])
                        edge.set_information(Inf)
                        # The robust kernel should be in accordance with 
                        # the 2d chi^2 distribution
                        edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))

                        # Here is what decides if its in the opt or not
                        edge.set_level(0) # if inlier_mask[k] else 1)

                        edges_per_obj.append(edge)
                        optimizer.add_edge(edge)

                    edges[view_id][obj_id] = edges_per_obj

         
        # Robust optimization with inlier/outlier classification from chi2 test
        if self.sfm_mode or self.slam_mode and not curr_only:
            its = [10, 10, 40, 40]
        else: # Single-view and SLAM (curr_only) mode with curr_only true or false
            its = [10] * 4 
            
        # Initial edge classification
        if self.opt_init_with_outliers and curr_only:
            # If the camera pose is bad, we want to optimize a bit with all the meas
            # to fix it instead of casting out outliers right away.
            num_good = len(optimizer.edges())
        else:
            num_good = 0
            for view_id, edges_curr in edges.items():
                for obj_id, es_in_obj in edges_curr.items():
                    for i, e in enumerate(es_in_obj):
                        e.compute_error() # Was not in last opt, need to compute.
                        chi2 = e.chi2()
                        if chi2 > 5.991:
                            e.set_level(1)
                            self.detections[view_id][obj_id]["inliers"][i] = False
                        else:
                            num_good += 1
                            e.set_level(0)
                            self.detections[view_id][obj_id]["inliers"][i] = True
        
        for it in range(len(its)):
            if len(optimizer.edges()) < 4 or num_good < 4:
                print(("CURR_OPT" if curr_only else "GLOBAL_OPT") + ": Not enough edges, quitting")
                break
            # Init opt with level==0, so only edges with level==0 will be included
            optimizer.initialize_optimization(0)
            optimizer.set_verbose(False)
            optimizer.optimize(its[it])
            
            # Reclassify inliers/outliers
            num_good = 0
            for view_id, edges_curr in edges.items():
                for obj_id, es_in_obj in edges_curr.items():
                    assert len(self.detections[view_id][obj_id]["inliers"]) == len(es_in_obj)
                    for i, e in enumerate(es_in_obj):
                        if not self.detections[view_id][obj_id]["inliers"][i]:
                            e.compute_error() # Was not in last opt, need to compute.
                        chi2 = e.chi2()
                        if chi2 > 5.991:
                            e.set_level(1)
                            self.detections[view_id][obj_id]["inliers"][i] = False
                        else:
                            num_good += 1
                            e.set_level(0)
                            self.detections[view_id][obj_id]["inliers"][i] = True
                    
                        # Get rid of robust kernel half-way through.
                        if it == max(1,len(its)//2):
                            e.set_robust_kernel(None)

        # Recover the optimized data
        for view_id, cam_vert in camera_verts.items():
            self.cam_poses[view_id] = cam_vert.estimate().matrix()[:3,:4]
        if not curr_only:
            for obj_id, obj_vert in object_verts.items():
                self.obj_poses[obj_id] = obj_vert.estimate().matrix()[:3,:4]
                # Check for bad object pose
                if view_id_curr in self.cam_poses.keys():
                    p_OinC = self.cam_poses[view_id_curr][:3,:3] @ self.obj_poses[obj_id][:3,3] \
                            + self.cam_poses[view_id_curr][:3,3]
                    if p_OinC[2] < 0.5*self.mesh_db[obj_id]["diameter"]:
                        print(f"OBJ {obj_id} is behind current camera. Removing.")
                        print(self.obj_poses[obj_id])
                        self.remove_obj(obj_id)

        # Remove objects with too few inliers
        obj_ids = list(self.obj_poses.keys()) # some may have been removed, so redo this here
        for obj_id in obj_ids:
            # Require a minimum number of inliers between all detections (6 typically).
            min_num_inliers = 3 if self.obj_num_dets[obj_id] < 3 else 6
            num_inliers = self.obj_num_inliers(obj_id)
            # Check if object needs to be removed outright.
            # We hope that it can be re-initialized later.
            # We remove if there are not enough inliers for a
            # few views in a row.
            print(f"ROBUST CHECK: obj_id {obj_id} num_inliers {num_inliers} "
                  f"thresh {min_num_inliers}")
            if num_inliers < min_num_inliers:
                print(f"ROBUST CHECK: object {obj_id} needs to be REMOVED "
                      f"({num_inliers} inliers between "
                      f"{self.obj_num_det_kps[obj_id]} detected keypoints, "
                      f"but {min_num_inliers} required)")
                self.remove_obj(obj_id)
    
    # Estimate camera pose in case of failure of __estimate_camera_pose
    def __backup_estimate_camera_pose(self, view_id, obj_ids_, bboxes):
        assert len(self.view_ids) > 0
        assert view_id not in self.view_ids
        assert view_id not in self.cam_poses.keys()

        # First, try to do a PnP for the bbox centers to the 3D bbox centers.
        # 3D bbox assumed to be aligned with object origin
        print("BACKUP_ESTIMATE_CAMERA_POSE: Attempting bbox centroid PnP")
        obj_ids = []
        bbox_centroids = [] # uv O in image (pixels)
        obj_centers = [] # p_OinG
        for i, obj_id in enumerate(obj_ids_):
            if obj_id in self.obj_poses.keys():
                obj_ids.append(obj_id)
                bbox_centroids.append(0.5 * (bboxes[i,:2] + bboxes[i,2:]))
                obj_centers.append(self.obj_poses[obj_id][:3,3])
        ret_pnp = None
        if len(bbox_centroids) > 0:
            bbox_centroids = np.stack(bbox_centroids)
            obj_centers = np.stack(obj_centers)
            ret_pnp = pnp(obj_centers, bbox_centroids, self.cam_K[view_id])
        if ret_pnp is not None:
            print("BACKUP_ESTIMATE_CAMERA_POSE: Bbox centroid PnP SUCCESS!")
            self.cam_poses[view_id] = ret_pnp[0]
        else:

            print("BACKUP_ESTIMATE_CAMERA_POSE: Bbox centroid PnP failed. Using const. vel. model.")
            # Failed. Try constant velocity model
            if len(self.view_ids) > 1:
                T_GtoC1 = np.eye(4)
                T_GtoC1[:3,:] = self.cam_poses[self.view_ids[-2]][:3,:]
                T_GtoC2 = np.eye(4)
                T_GtoC2[:3,:] = self.cam_poses[self.view_ids[-1]][:3,:]
                T_C1toC2 = T_GtoC2 @ utils.invert_SE3(T_GtoC1)
                self.cam_poses[view_id] = T_C1toC2 @ T_GtoC2
            else:
                # Just set equal to last pose (TODO any other option?)
                self.cam_poses[view_id] = self.cam_poses[self.view_ids[-1]]
        
        # We will always get a pose from this method, no matter how bad.
        self.view_ids.append(view_id)

    def __estimate_camera_pose(self, view_id, min_num_inliers=4):
        """
        Estimate the current camera pose with a RANSAC loop which finds the
        most likely camera pose from pairs of individual object pose estimates
        between the last and current frame.

        We transform the object points from the global frame to the current 
        frame with all current camera pose hypotheses that result from solving
        for the current camera pose given the current single-view pose results and the
        global object pose estimates, and taking the camera pose that has the most inliers
        of keypoints in the image plane.
        """
        assert self.num_views_processed() > 0
        assert view_id not in self.cam_poses.keys()
        
        t0 = time()
        curr_det = self.detections[view_id]
        # Match the object ids
        obj_ids = []
        for obj_id in curr_det.keys():
            # We need the 3D estimate of the object, and a good PnP pose from the current frame.
            if curr_det.get(obj_id,{}).get("pose") is not None and obj_id in self.obj_poses:
                obj_ids.append(obj_id)
        print("RANSAC obj_ids:", obj_ids)
        if len(obj_ids) == 0:
            print("RANSAC: Not enough object candidates. Quitting.")
            return

        Ts_GtoO = np.stack([utils.invert_SE3(self.obj_poses[obj_id]) for obj_id in obj_ids])
        Ts_OtoG = np.zeros((len(obj_ids),4,4), dtype=np.float32)
        for j in range(len(obj_ids)):
            Ts_OtoG[j,:3,:] = self.obj_poses[obj_ids[j]][:3,:]
            Ts_OtoG[j,3,3] = 1
        Ts_OtoC_pnp = np.stack([curr_det[obj_id]["pose"] for obj_id in obj_ids])
        Ts_CtoO_pnp = np.stack([utils.invert_SE3(curr_det[obj_id]["pose"]) for obj_id in obj_ids])
        Ts_hypoth_GtoC = Ts_OtoC_pnp @ Ts_GtoO
 
        # Add the hypothesis from bbox PnP. It may be better than the object PnP in some cases.
        #bbox_centroids = [] # uv O in image (pixels)
        #obj_centers = [] # p_OinG
        #for obj_id in obj_ids:
        #    bbox = curr_det[obj_id]["bbox"]
        #    bbox_centroids.append(0.5 * (bbox[:2] + bbox[2:]))
        #    obj_centers.append(self.obj_poses[obj_id][:3,3])
        #ret_pnp = None
        #if len(bbox_centroids) > 0:
        #    bbox_centroids = np.stack(bbox_centroids)
        #    obj_centers = np.stack(obj_centers)
        #    ret_pnp = pnp(obj_centers, bbox_centroids, self.cam_K[view_id])
        #if ret_pnp is not None:
        #    T_GtoC_bbpnp = np.eye(4)
        #    T_GtoC_bbpnp[:3,:] = ret_pnp[0][:3,:]
        #    Ts_hypoth_GtoC = np.concatenate((Ts_hypoth_GtoC,T_GtoC_bbpnp[None,:,:]), axis=0)

        Ts_OtoC_hypoth = Ts_hypoth_GtoC[:,None,:,:] @ Ts_OtoG[None,:,:,:] # num_hypoth x num_obj
        T_GtoC_best = None
        best_num_inliers = -1
        for i in range(Ts_OtoC_hypoth.shape[0]):
            num_inliers_i = 0
            for j in range(len(obj_ids)):
                obj_id = obj_ids[j]
                if np.count_nonzero(curr_det[obj_id]["inliers"]) > 0:
                    # Project the object points of obj_ids[j] into the 
                    # camera using Ts_hypoth_GtoC[i]
                    p_FinC = utils.transform_pts(Ts_OtoC_hypoth[i,j],
                            curr_det[obj_id]["model_kp"][curr_det[obj_id]["inliers"]])
                    uv_proj = p_FinC @ curr_det[obj_id]["K"].T
                    d_pos_mask = uv_proj[:,2] > 0
                    uv_proj = (uv_proj[:,:2] / uv_proj[:,2:3])[d_pos_mask]
                    if uv_proj.shape[0] > 0:
                        # Calculate chi2. Cov may be None if not using predicted covariance.
                        # Note that UVs are in normalized device coords, so manual
                        # covariance needs to be scaled properly for values in [-1,1]
                        uv = curr_det[obj_id]["uv_pred"][curr_det[obj_id]["inliers"]][d_pos_mask]
                        cov = curr_det[obj_id]["cov_pred"]
                        res = uv - uv_proj
                        if cov is not None:
                            cov = cov[curr_det[obj_id]["inliers"]][d_pos_mask]
                            # Ensure invertible
                            cov[:,[0,1],[0,1]] = np.maximum(cov[:,[0,1],[0,1]], 1e-4)
                            inf = np.linalg.inv(cov)
                            assert not np.any(np.isnan(inf)), \
                                    f"NaN in information matrix!\n\nInf = {inf}\nCov = {cov}"
                        else:
                            inf = np.zeros((res.shape[0],2,2), dtype=np.float32)
                            sigma = self.manual_kp_std
                            inf[:,[0,1],[0,1]] = 1 / sigma**2

                        chi2 = np.squeeze(res[:,None,:] @ inf @ res[:,:,None], -1)
                        #print("Chi2:", chi2)
                        # inliers have a chi2 less than 5.991 for 2 DoF in 95% confidence.
                        num_inliers_i += np.count_nonzero(chi2 <= 5.991)

            if num_inliers_i >= min_num_inliers and num_inliers_i > best_num_inliers:
                T_GtoC_best = Ts_hypoth_GtoC[i]
                best_num_inliers = num_inliers_i
        print(f"RANSAC best_num_inliers={best_num_inliers} (t={1000 * (time()-t0):.3f} ms)\n")
        return T_GtoC_best
    
    # prior_dets will be a dict of tensors [num_kp h w] tensor from utils.make_prior_kp_input
    # for each object that it is available. obj_ids may have objecrt IDs that are not in prior_dets
    # obviously.
    def __run_kp_model(self, view_id, img, K, obj_ids, bboxes, 
            model_kps, model_kps_masks,  kp_masks_gt=None, uv_gt=None, prior_dets=None):
        # If no prior, this will be left as zeros
        priors_np = np.zeros([len(obj_ids), kp_config.num_kp()]+self.pred_res, dtype=np.float32)
        # K for each bbox
        K_bbox_np = np.zeros([len(obj_ids),3,3], dtype=np.float32)
        for k, obj_id in enumerate(obj_ids):
            if prior_dets is not None and obj_id in prior_dets.keys():
                priors_np[k] = prior_dets[obj_id]
            K_bbox_np[k] = utils.fix_K_for_bbox_ndc(K, bboxes[k])

        cov_uv = None
        if not self.debug_gt_kp:
            with torch.no_grad():
                # Run the network on batch of object detections (still single image)
                img_th = torch.tensor(img.transpose((2,0,1)).astype(np.float32)/255)[None,...]
                # Bboxes and priors are fed in as list
                bboxes_th = [torch.tensor(bboxes, dtype=torch.float32)]
                priors_th = [torch.tensor(priors_np)]
                if torch.cuda.is_available():
                    img_th = img_th.cuda()
                    bboxes_th, priors_th = [bboxes_th[0].cuda()], [priors_th[0].cuda()]
                pred = self.model(img_th, bboxes_th, priors_th)
                exp_uv = pred["uv"].cpu().numpy()
                #print(pred["kp_mask"].cpu().numpy())
                kp_masks = (pred["kp_mask"].cpu().numpy() > 0.3) & model_kps_masks
                # Ignore keypoints that are too near the boundaries.
                # They are probably bad.
                kp_masks = kp_masks & (np.min(exp_uv,-1) > -self.bbox_thresh) \
                        & (np.max(exp_uv,-1) < self.bbox_thresh)

                if not self.no_network_cov:
                    cov_uv = pred["cov"].cpu().numpy()
                    # Mask out keypoints with min stdev (between x and y direction) of
                    # percent_thresh% of the bbox dimension or greater. We can do this without 
                    # transfering to pixel coords since the bbox dims in ndc are w,h=2,2
                    std = np.sqrt(cov_uv[...,[0,1],[0,1]])
                    percent_thresh = self.kp_var_thresh
                    kp_masks = kp_masks & np.all(std < 2*percent_thresh, axis=-1)
        else:
            assert kp_masks_gt is not None
            assert uv_gt is not None
            kp_masks = kp_masks_gt

        # Now compute the poses through PnP
        ret_data = []
        for k, obj_id in enumerate(obj_ids):
            kp_mask = kp_masks[k]
            
            if not self.debug_gt_kp:
                uv_pred = exp_uv[k][kp_mask].astype(np.float64)
            else:
                # Add some idealized noise to GT uv for debugging
                uv_pred = uv_gt[k][kp_mask].astype(np.float64)
                uv_pred += np.random.normal(scale=0.01, size=uv_pred.shape)
            cov_pred = None
            if cov_uv is not None:
                cov_pred = cov_uv[k][kp_mask]

            # Grab the 3D points in the model's frame.
            kp_model = model_kps[k][kp_mask].astype(np.float64)
            K_kp = K_bbox_np[k].astype(np.float64)

            pose = None
            # Ignore inliers/outliers from PnP since we perform
            # extensive chi2 testing during optimization.
            inliers = np.ones(uv_pred.shape[0], dtype=np.bool)
            ret_pnp = pnp(kp_model, uv_pred, K_kp)
            if ret_pnp is not None:
                T_OtoC, inliers_pnp = ret_pnp
                if T_OtoC[2,3] > 0.5*self.mesh_db[obj_id]["diameter"] \
                        and np.count_nonzero(inliers_pnp) >= 4:
                    pose = T_OtoC
                    if pose.shape[0] == 3:
                        pose = np.concatenate((pose,np.eye(4)[3:4,:]), axis=0)

            self.obj_num_dets[obj_id] += 1
            self.obj_num_det_kps[obj_id] += uv_pred.shape[0]
            ret_data.append({
                "pose": pose,
                "inliers": inliers, 
                "kp_mask": kp_mask, # Mask for all kp in kp_config
                "model_kp": kp_model,
                "uv_gt": uv_gt,
                "uv_pred": uv_pred,
                "cov_pred": cov_pred,
                "K": K_kp,
                "score": 0.0 if inliers.size==0 else inliers.astype(np.float32).mean(),
            })

        return ret_data
