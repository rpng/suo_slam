import os
import gc
import cv2
import math
import json
import torch
import shutil
import numpy as np
import pickle as pkl
from time import time, strftime
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

cv2.setNumThreads(0) 

from ..labeling import kp_config
from ..models.pkpnet import mesh_grid

def device_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time()

def log_so3(R):
    """
    SO(3) matrix logarithm.
    This definition was taken from "Lie Groups for 2D and 3D Transformations" 
    by Ethan Eade equation 17 & 18.

    :param[in] R [3,3] numpy array of SO(3) rotation matrix
    :return [3,] numpy array in the se(3) space [omegax, omegay, omegaz]
    """

    assert R.shape == (3,3), "Invalid rotation matrix shape"

    # Magnitude of the skew elements (handle edge case where we sometimes have a>1...)
    a = 0.5 * (np.trace(R) - 1)
    theta = math.acos(1) if a > 1 else (math.acos(-1) if a < -1 else math.acos(a))
    # Handle small angle values
    D = 0.5 if theta < 1e-12 else theta/(2*math.sin(theta))
    # Calculate the skew symetric matrix
    w_x = D * (R - R.T)
    # Check if we are near the identity
    if np.allclose(R, np.eye(3)):
        return np.zeros((3,), dtype=R.dtype)
    else:
        return np.array([w_x[2,1], w_x[0,2], w_x[1,0]], dtype=R.dtype)

def euler2R(euler):
    # Assume euler = {gamma, beta, alpha} (degrees)
    g, b, a = np.deg2rad(euler).astype(np.float64)
    cosa = np.cos(a)
    cosb = np.cos(b)
    cosg = np.cos(g)
    sina = np.sin(a)
    sinb = np.sin(b)
    sing = np.sin(g)
    R = np.array([[cosa*cosb, cosa*sinb*sing - sina*cosg, cosa*sinb*cosg + sina*sing],
                  [sina*cosb, sina*sinb*sing + cosa*cosg, sina*sinb*cosg - cosa*sing],
                  [-sinb, cosb*sing, cosb*cosg]], dtype=np.float64)
    return R

# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-4

# Get the Euler angles in degrees
def rot2euler(R):
    assert is_rotation_matrix(R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return 180.0 * np.array([x, y, z]) / np.pi

def variance_loss(uv, prob, mask):
    """

    Args:
          uv is in ndc: (batch_size, 10, 2)
          prob is the softmaxed keypointnet output (batch_size, 10, 128, 128)
    """    
    # NOTE: Assume prob already softmaxed
    vh, vw = prob.shape[2], prob.shape[3]
    xx, yy = mesh_grid(vh, vw, prob.device)

    xy = torch.stack([xx, yy], 2)
    sh = xy.shape

    xy = torch.reshape(xy, [1, 1, sh[0], sh[1], 2])
    sh = uv.shape

    uv = torch.reshape(uv, [sh[0], sh[1], 1, 1, 2])
    diff = torch.sum(torch.square(uv-xy), 4)
    diff *= prob

    return torch.mean(torch.sum(diff, [2,3])[mask])

def mle_loss(uv_hat, uv, cov):
    """
    Calculate the MLE loss 
    :param uv_hat: reprojected uv shape(..., 2)
    :param uv:     UV measurements shape(..., 2)
    :param cov:    2x2 covariance matrix for uv shape(..., 2, 2)
    """
    res = uv - uv_hat

    # Ensure cov invertable
    cov[...,0,0] += 1e-6
    cov[...,1,1] += 1e-6

    # Ensure stability with cholesky decomp 
    # (NOTE can cause crash https://github.com/pytorch/pytorch/pull/50957)
    #S = torch.cholesky(torch.inverse(cov))
    #resS = torch.squeeze(torch.unsqueeze(res, -2) @ S, -2)
    #loss_mv = resS.square().sum(-1).mean()
    loss_mv = (res.unsqueeze(-2) @ torch.inverse(cov) @ res.unsqueeze(-1)).mean()

    # Add log determinant of cov to complete MLE loss
    # Note that the 0.5 multiplication happens in the main loss function
    def logdet_2x2(x):
        det = x[...,0,0]*x[...,1,1] - x[...,1,0]*x[...,0,1]
        ld = torch.log(torch.maximum(det, 1e-12*torch.ones_like(det)))
        
        # NOTE torch.logdet can fail because of a MAGMA issue.
        #ld_ = torch.logdet(x)
        #assert torch.allclose(ld, ld_)
        return ld

    loss_cov = logdet_2x2(cov).mean()

    return loss_mv, loss_cov

# Args
# pred:   dict output of PgNet
# target: (torch.float32) [B K 2] GT keypoints in normalized device 
#         coordinates (2*u/w-1, 1-2*v/h) for each image pixel coordinate (u, v).
# mask:   (torch.bool) [B K] Mask for which keypoint channels are valid based on object.
# 
# Returns
# uv_loss, var_loss, mask_loss (scalar torch.float32) losses for l2 coordinate regression and 
#                   2d probability grid variance minimization, and BCE for the predicted mask.
def kp_loss(pred, target, mask=None, epoch=0):
    if mask is None:
        mask = torch.ones(target.shape[:-1], dtype=torch.bool, device=target.device)

    exp_uv = pred["uv"]
    prob = pred["prob"]
    
    if mask.count_nonzero() == 0:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, zero, zero

    if "cov" in pred.keys():
        uv_loss, var_loss = mle_loss(exp_uv[mask], target[mask], pred["cov"][mask])
    else:
        # L2 loss
        res = target - exp_uv
        uv_loss = res.square().sum(-1)[mask].mean()

        # Minimize variance.
        var_loss = variance_loss(exp_uv, prob, mask)
    
    # Binary cross entropy for mask that predicts whether keypoint is valid or not
    kp_mask_pred = pred["kp_mask"] # Note this has already been sigmoided
    bce_mask_loss = F.binary_cross_entropy(kp_mask_pred, mask.to(torch.float))

    return uv_loss, var_loss, bce_mask_loss

# If cov is not None, draw the 2D Gaussian ellipse
def draw_points(rgb, xy, cols, cov=None, labels=None, ndc=True, rad=4):
    """Draws keypoints onto an input image.

    Args:
    rgb: Input image to be modified.
    xy: [n x 2] matrix of 2D locations.
    cols: A list of colors for the keypoints.
    """
    
    if labels is not None:
        assert len(labels) == len(xy)
    assert rgb.dtype == np.uint8
    assert len(rgb.shape) == 3
    assert rgb.shape[-1] == 3
    vh, vw = rgb.shape[0], rgb.shape[1]
    assert xy.shape[0] == cols.shape[0]
    assert len(xy.shape) == 2 # [K 2] (u v)
    assert len(cols.shape) == 2, f"Incorrect color shape: {cols.shape}\n\n{cols}" # [K 3] (BGR)
    if cov is not None:
        #print("COV SHAPE:", cov.shape)
        #print("XY SHAPE:", xy.shape)
        assert len(cov.shape) == 3
        assert cov.shape[0] == xy.shape[0]
        assert cov.shape[1] == 2
        assert cov.shape[2] == 2

    for j in range(len(cols)):
        x, y = xy[j, :2]
        if ndc:
            x = (min(max(x, -1), 1) * vw / 2 + vw / 2) - 0.5
            y = vh - 0.5 - (min(max(y, -1), 1) * vh / 2 + vh / 2)
        x = int(round(x))
        y = int(round(y))
        #print(x,y)
        if x < 0 or y < 0 or x >= vw or y >= vh:
            continue
        col = cols[j]
        if isinstance(col, np.ndarray):
            col = col.tolist()
        assert type(col) == list, f"Invalid color type {type(col)} expected list"
        # Note to reverse color for OpenCV BGR
        rgb = cv2.circle(rgb, (x,y), int(round(1.3*rad)), [0,0,0], -1)
        rgb = cv2.circle(rgb, (x,y), int(round(rad)), col, -1)
        # Draw covariance ellipse
        if cov is not None:
            if ndc:
                assert False
                # Cov has to be scaled up to pixels instead of normalized device coords
                S = np.array([[vw/2,0.0],[0.0,vh/2]])
                cov_i = S @ cov[j] @ S.T
            else:
                # Cov should already be scaled
                cov_i = cov[j]
            lamb, v = np.linalg.eig(cov_i)
            # Ellipse angle is based on direction of first eigenvector
            angle = np.arctan2(v[1,0], v[0,0])
            s = 1.0/3 # Draw s*3-sigma ellipse
            axes_length = (int(round(s*2*np.sqrt(5.991*lamb[0]))), 
                           int(round(s*2*np.sqrt(5.991*lamb[1]))))
            rgb = cv2.ellipse(rgb, (x,y), axes_length, 
                    180*angle/np.pi, 0, 360, col, 2) 
        
        if labels is not None:
            rgb = cv2.putText(rgb, labels[j], (x+10, y), cv2.FONT_HERSHEY_PLAIN, 1, 
                    (255,255,255), 1, cv2.LINE_AA)

def bbox_color(obj_id, num_obj=30):
    cols = (255*plt.cm.get_cmap("gist_rainbow")
            (np.linspace(0, 1.0, num_obj))[:, :3][:,::-1]).astype(np.int)
    return cols[obj_id-1].tolist()

# bboxes should contain 5 numbers. obj_id, then xyxy bbox. All as np.int
# poses is dict mapping obj_id to T_OtoC 
def make_kp_viz(image, kp_pred, kp_mask, kp_gt=None, bbox_gt=None, bbox_pred=None, 
        cov=None, prior=None, ndc=True, poses=None, K=None, mesh_db=None, 
        heatmaps=None, rad=8):
    rgb = np.copy(np.ascontiguousarray(image))
    if prior is not None:
        assert heatmaps is None, "Cannot viz priors and heatmaps at once."
        if prior.dtype == np.uint8:
            prior = prior.astype(np.float32) / 255
        prior = np.clip(np.sum(
                prior.transpose((1,2,0))[...,None] * kp_config.kp_colors()[None,None,...], 
                axis=2),0,255).astype(np.uint8)
    elif heatmaps is not None:
        prior = np.clip(np.sum(
                heatmaps[...,None] * kp_config.kp_colors()[None,None,...], 
                axis=2),0,255).astype(np.uint8)

    if bbox_pred is None and bbox_gt is not None:
        bbox_pred = bbox_gt
        
    # Draw pose projected points
    if poses is not None:
        assert K is not None and mesh_db is not None, "Pose provided but not K or mesh_db"
        # TODO might wanty to draw objects in order of farthest to closest
        K_ = torch.tensor(K)
        if torch.cuda.is_available():
            K_ = K_.cuda()
        # SOrt from farthest to closest
        poses_sorted = sorted(list(poses.items()), key=lambda p: -p[1][2,3])
        for obj_id, T_OtoC in poses_sorted:
            # Project points into camera frame
            pts_in_O = mesh_db[obj_id]["points"]
            # Subsample points for speed
            #num_samples = 666
            #if pts_in_O.shape[0] > num_samples:
            #    pts_in_O = pts_in_O[torch.randperm(pts_in_O.shape[0], dtype=torch.long, 
            #            device=pts_in_O.device)[:num_samples]]
            pts_in_C = transform_pts(torch.tensor(T_OtoC.astype(np.float32)).to(pts_in_O.device),
                                     pts_in_O)
            uvd = pts_in_C @ K_.T
            uv = uvd[:,:2] / uvd[:,2:3]
            # Round to nearest pixel
            uv = (uv + 0.5).to(torch.int)
            # Delete out-of-bounds ones
            in_bounds_rows = (uv[:,1] >= 0) & (uv[:,1] < rgb.shape[0])
            in_bounds_cols = (uv[:,0] >= 0) & (uv[:,0] < rgb.shape[1])
            uv = uv[in_bounds_rows & in_bounds_cols]
            # Rasterize
            uv = uv.cpu().numpy()
            if uv.shape[0] > 0:
                obj_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                obj_mask[uv[:,1], uv[:,0]] = 255
                # Fill in mask a bit
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                obj_mask = cv2.dilate(obj_mask, kernel, iterations=1)
                #obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)
                obj_mask = obj_mask > 0
                rgb[obj_mask] = bbox_color(obj_id)

    # Draw bboxes
    if bbox_pred is not None:
        for i in range(len(bbox_pred)):
            obj_id, x1, y1, x2, y2 = bbox_pred[i]
            cv2.rectangle(rgb, (x1,y1), (x2,y2), bbox_color(obj_id), 3)
            rgb = cv2.putText(rgb, f"obj_{obj_id}", 
                    (x1+8, y1-8), cv2.FONT_HERSHEY_PLAIN, 1, 
                    bbox_color(obj_id), 2, cv2.LINE_AA)

    if kp_gt is not None:
        rgb_gt = np.copy(rgb)
        if bbox_gt is not None:
            for i in range(len(bbox_gt)):
                obj_id, x1, y1, x2, y2 = bbox_gt[i]
                cv2.rectangle(rgb_gt, (x1,y1), (x2,y2), bbox_color(obj_id), 3)
                rgb_gt = cv2.putText(rgb_gt, f"obj_{obj_id}", 
                        (x1+8, y1-8), cv2.FONT_HERSHEY_PLAIN, 1, 
                        bbox_color(obj_id), 2, cv2.LINE_AA)

    for i in range(kp_pred.shape[0]):
        uv = kp_pred[i][kp_mask[i]]
        cov_i = cov[i][kp_mask[i]] if cov is not None else None
        cols = kp_config.kp_colors()[kp_mask[i]]
        kp_labels = None #np.array(kp_config.kp_list)[kp_mask[i]]
        draw_points(rgb, uv, cols, cov=cov_i, labels=kp_labels, ndc=ndc, rad=rad)
        if kp_gt is not None:
            uv_gt = kp_gt[i][kp_mask[i]]
            draw_points(rgb_gt, uv_gt, cols, labels=kp_labels, ndc=ndc)

    def blend(img):
        if prior is not None and prior.size > 0:
            prior_prob = cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY).astype(np.float32)[...,None] / 255
            # Blend based on the prior prob
            a = 1 - prior_prob
            b = prior_prob
            return (a * img + b * prior).astype(np.uint8)
        else:
            return img
    if kp_gt is not None:
        return np.concatenate((blend(rgb_gt), blend(rgb)), axis=1)
    else:
        return blend(rgb)

def gaussian_2d(sigma):
    assert sigma % 2 == 1, "Sigma must be odd"
    gauss = np.zeros((sigma,sigma), dtype=np.float32)
    gauss[sigma//2,sigma//2] = 1
    gauss = cv2.GaussianBlur(gauss, (sigma,sigma), 0)
    return gauss / np.max(gauss)

# NOTE: that pt is [x,y] UV pixel coordinate
def draw_gaussian_2d(img, pt, sigma=15):
    assert len(img.shape) == 2
    assert img.dtype == np.float32

    tmpSize = int(np.math.ceil(3 * sigma))
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img

    size = 2 * tmpSize + 1
    g = gaussian_2d(size)

    g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

    img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0])]

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

# Make the tensor (float32 numpy array) of the prior predictions for all kp in kp_uv.
# Invalid kp are filled with zeros.
# == Args ==============================================================
# kp_uv: [N 2] np.array normalized device coordinates for N keypoints
# kp_uv_mask: [N] np.bool array of whether or not the kp value is valid
# img_shape: list-like of len 2 for the image shape (height, width)
# ndc: bool if True, consider the UVs as normalized device coords. If false,
#      consider them to be the UV image coordinates.
# == Returns =========================================================
# x: [N height width] np.float32 np.array of the prior predictions as 
#                     Gaussians with a high sigma from gaussian blurring.
def make_prior_kp_input(kp_uv, kp_uv_mask, img_shape, ndc=True):
    n = kp_uv.shape[0]
    x = np.zeros((n, img_shape[0], img_shape[1]), dtype=np.float32)
    vh, vw = img_shape
    for i in range(n):
        if kp_uv_mask[i] and np.all(np.isfinite(kp_uv[i, :2])):
            u, v = kp_uv[i, :2]
            if ndc:
                u = (min(max(u, -1), 1) * vw / 2 + vw / 2) - 0.5
                v = vh - 0.5 - (min(max(v, -1), 1) * vh / 2 + vh / 2)
            pt = (int(round(u)), int(round(v)))
            # Draw Gaussian and blur it to account for noise
            x[i] = draw_gaussian_2d(x[i], pt)
    return x

# Create the camera matrix that projects a 3d point in camera frame x 
# onto the bbox's image plane in normalized device coordinates in [-1,1]
# so that the raw keypoint network predictions can be used as UV coordinates.
def fix_K_for_bbox_ndc(K_, bbox):
    x1, y1, x2, y2 = bbox
    x, y, w, h = x1, y1, x2-x1, y2-y1
    K = np.copy(K_)
    # Fix the K matrix for the operations on UV
    duv = np.array([x,y], dtype=np.float64)
    T = np.eye(3)
    T[:2,2] = -duv
    S = np.eye(3)
    S[0,:] *= 2.0/w
    S[1,:] *= -2.0/h
    S[0,2] -= 1
    S[1,2] += 1
    return S @ T @ K

def invert_SE3(T):
    Tinv = np.eye(4)
    Tinv[:3,:3] = T[:3,:3].T
    Tinv[:3,3] = -T[:3,:3].T @ T[:3,3]
    return Tinv

def transpose_mat(A):
    # Transpose matrix with arbitrary batch dimensions.
    t_inds = [i for i in range(len(A.shape[:-2]))] + [-1,-2]
    if type(A) == np.ndarray:
        return A.transpose(t_inds)
    elif type(A) == torch.Tensor:
        return A.permute(t_inds)
    else:
        raise ValueError("transpose_mat: Unsupported type {type(A)}")

'''
Transform either numpy array or torch tensor of points
T: shape is of shape [b0, b1, ..., b(N), (3 or 4), 4], [b0, b1, ..., b(N-1), (3 or 4), 4], ..., 
   or [b0, (3 or 4), 4] for batch dimensions b(i). In the case where batch dims are missing,
   they are filled with singleton dimenions and broadcasted to all dimensions of `pts`.
pts: [b0, b1, ..., 3]
'''
def transform_pts(T, pts):
    assert len(T.shape)-1 <= len(pts.shape), \
            f"T ({T.shape}) has more batch dimensions than pts ({pts.shape})!"
    while len(T.shape)-1 < len(pts.shape):
        T = T[...,None,:,:]
    ret = pts @ transpose_mat(T[..., :3, :3]) + T[..., :3, 3]
    return ret.reshape(pts.shape)

def debug_print_tensor_sizes():
    print("\n====================================>\nTensors in scope:")
    num = 0
    gb = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #if type(obj) != torch.nn.parameter.Parameter and obj.numel() > 256:
                #print(type(obj), obj.shape)
                num += 1
                if obj.dtype == torch.float32:
                    gb += obj.numel() * 4 * 1e-9
                else:
                    assert False
        except:
            pass
    print(f"Found {num} large tensors in scope totalling {gb:.3f} GB of GPU memory")
    print("<====================================")

def load_posecnn_results(bop_root):
    results_path = os.path.join(bop_root, 'saved_detections/ycbv_posecnn.pkl')
    #results = pkl.loads(results_path.read_bytes())
    with open(results_path, 'rb') as results_fp:
        results = pkl.load(results_fp)

    data = {
        "scene_ids": [],
        "view_ids": [],
        "scores": [],
        "obj_ids": [], 
        "poses": [],
        "bboxes": [],
    }

    #f_offsets = os.path.join(bop_root, 'bop_datasets/ycbv/offsets.txt')
    f_offsets = os.path.join(bop_root, 'ycbv/offsets.txt')
    with open(f_offsets, 'r') as offset_fp:
        l_offsets = offset_fp.read().strip().split('\n')
    ycb_offsets = dict()
    for l_n in l_offsets:
        obj_id, offset = l_n[:2], l_n[3:]
        obj_id = int(obj_id)
        offset = np.array(json.loads(offset)) #* 0.001
        ycb_offsets[obj_id] = offset

    def mat_from_qt(qt):
        wxyz = qt[:4].copy().tolist()
        xyzw = [*wxyz[1:], wxyz[0]]
        t = qt[4:].copy()
        R = Rotation.as_matrix(Rotation.from_quat(xyzw))
        return np.concatenate((R,t[:,None]), axis=1)

    for scene_view_str, result in results.items():
        scene_id, view_id = scene_view_str.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        n_dets = result['rois'].shape[0]
        for n in range(n_dets):
            obj_id = result['rois'][:, 1].astype(np.int)[n]
            data["scene_ids"].append(scene_id)
            data["view_ids"].append(view_id)
            # Same as object_id, not a real score
            data["scores"].append(result['rois'][n, 1])
            data["obj_ids"].append(obj_id)
            data["bboxes"].append(result['rois'][n, 2:6])
            pose = mat_from_qt(result['poses'][n])
            pose[:3,3] *= 1000 # m to mm
            offset = ycb_offsets[obj_id]
            # Translate the points to original YCB model frame
            # to use the PoseCNN results
            T_orig2bop = np.eye(4)
            T_orig2bop[:3,3] = -offset
            pose = pose @ T_orig2bop
            data["poses"].append(pose)
    return data

# TODO tless results
def load_pix2pose_results(bop_root):
    results_path = os.path.join(bop_root, 'saved_detections/tless_pix2pose_retinanet_siso_top1.pkl')
    with open(results_path, 'rb') as results_fp:
        results = pkl.load(results_fp)

    data = {
        "scene_ids": [],
        "view_ids": [],
        "scores": [],
        "obj_ids": [], 
        "poses": [],
        "bboxes": [],
    }

    for scene_view_str, result in results.items():
        scene_id, view_id = scene_view_str.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        n_dets = result['rois'].shape[0]
        boxes = result['rois']
        new_boxes = boxes.copy()
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,3] = boxes[:,2]
        for n in range(n_dets):
            obj_id = result['rois'][:, 1].astype(np.int)[n]
            data["scene_ids"].append(scene_id)
            data["view_ids"].append(view_id)
            data["scores"].append(result['rois'][n, 1])
            data["obj_ids"].append(int(result['labels_txt'][n].split('_')[-1]))
            data["bboxes"].append(new_boxes[n].astype(np.float32))
            pose = result['poses'][n]
            pose[:3,3] *= 1000 # m to mm
            data["poses"].append(pose)
    return data

if __name__ == '__main__':
    data = load_posecnn_results("/media/nate/Elements/bop")
    #print(data["segmask_paths"])
