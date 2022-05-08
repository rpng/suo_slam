import torch
import numpy as np
from collections import defaultdict

from . import utils

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''
Sample N points from an MxK torch.Tensor or np.array where M>=N and K is the dim of the points
'''
def sample_pts(n, pts):
    if type(pts) == torch.Tensor:
        return torch.randperm(pts.shape[0], device=pts.device)[:n]
    elif type(pts) == np.ndarray:
        return pts[np.random.choice(range(pts.shape[0]), size=n, replace=False)]
    else:
        raise ValueError("transpose_mat: Unsupported type {type(A)}")

# Adapted from CosyPose repo
def compute_auc_posecnn(errors):
    if type(errors) == list:
        errors = np.array(errors, dtype=np.float32)
    errors = np.squeeze(errors)
    #assert np.all(np.isfinite(errors))
    # NOTE: change mm to m
    errors = 1e-3 * errors.copy()
    errors[errors > 0.1] = np.inf
    d = np.sort(errors)
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    if not (len(ids) > 0 and ids.sum() > 0):
        return 0
    d = d[ids]
    accuracy = accuracy[ids]
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids-1]) * mpre[ids]).sum() * 10
    return ap

class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, x, k=1):
        # Recursive average update for more numeric stability
        # Proof:
        # mu_n = (x1+...+x(n-1)+xn) / n
        #      = (n-1)/(n-1) * (x1+...+x(n-1)+xn) / n
        #      = [ (n-1)/(n-1)*(x1+...+x(n-1)) + (n-1)/(n-1)*xn] / n
        #      = [(n-1)*mu_(n-1) + xn] / n
        self.n += k
        self.avg = ((self.n-k)*self.avg + x) / self.n

    def average(self):
        return self.avg

# Keeps track of the errors on a per-class basis, then computes AUC 
class AddAucMeter:
    def __init__(self, obj_avg=False):
        self.err_map = defaultdict(list)
        self.obj_avg = obj_avg

    # obj_ids: list of int class IDs
    # errs: list of float errors
    def update(self, obj_ids, errs):
        for obj_id, err in zip(obj_ids, errs):
            self.err_map[obj_id].append(err)

    def average(self):
        n = len(self.err_map)
        # NOTE: CosyPose does the average of all AUC of ADD but
        # it seems that PoseCNN did the AUC of all the errors together.
        # Here we do the PoseCNN method to stay true to the original method.
        auc_sum = 0
        assert n > 0, "Called AucMeter.average without feeding any data!"
        auc_map = {}
        errs_tot = []
        for obj_id, errs in self.err_map.items():
            auc = compute_auc_posecnn(errs)
            auc_map[obj_id] = auc
            errs_tot += errs
            auc_sum += auc
        if self.obj_avg:
            return auc_sum / n, auc_map
        else:
            auc_tot = compute_auc_posecnn(errs_tot)
            return auc_tot, auc_map

class EvalMeter:
    def __init__(self, mesh_db, sample_n_points=None, d=0.1):
        self.mesh_db = mesh_db
        self.d = d
        self.sample_n_points = sample_n_points
        if sample_n_points is not None:
            assert type(sample_n_points) == int
            for obj_id in self.mesh_db.keys():
                assert sample_n_points <= self.mesh_db[obj_id]["points"], \
                        "Not enough points in mesh to sample {sample_n_points} points"
                if "points_sampled" not in self.mesh_db[obj_id].keys() \
                        or self.mesh_db[obj_id]["points_sampled"].shape[0] != sample_n_points:
                    self.mesh_db[obj_id]["points_sampled"] = sample_pts(sample_n_points, 
                            self.mesh_db[obj_id]["points"])

        # NOTE: We average the AUC over each object as in DeepIM and CosyPose instead
        # of taking the AUC of all objects together like the original PoseCNN did.
        # ADD AUC
        self.add_meter = AddAucMeter(obj_avg=True)

        # ADD-S AUC
        self.adds_meter = AddAucMeter(obj_avg=True)
        
        # ADD(-S) AUC
        self.add_maybe_s_meter = AddAucMeter(obj_avg=True)

    def update(self, obj_ids, poses_pred, poses_gt):
        
        if self.sample_n_points is not None:
            points = torch.stack([self.mesh_db[obj_id]["points_sampled"] for obj_id in obj_ids])
        else:
            assert len(obj_ids) == 1, "Please set sample_n_points if you want to eval in batch"
            points = self.mesh_db[obj_ids[0]]["points"][None,...] 
        
        poses_pred = self.__to_tensor(poses_pred, points.device).to(torch.float32)
        poses_gt = self.__to_tensor(poses_gt, points.device).to(torch.float32)

        # NOTE: Doesn't matter if we change to mm here or in compute_auc_posecnn
        #points = 1e-3 * points
        #poses_pred[...,:3,3] *= 1e-3
        #poses_gt[...,:3,3] *= 1e-3

        # Blindly use the ADD-S for this metric whether symmetric or not.
        pred_points = utils.transform_pts(poses_pred, points) # Pre-transform points
        gt_points = utils.transform_pts(poses_gt, points)
        dists_add = self.__dists_add(pred_points, gt_points)
        dists_adds = self.__dists_add_sym(pred_points, gt_points)

        # Use ADD for those that are not symmetric, fill in (-S) with work already done.
        is_sym = torch.tensor([self.mesh_db[obj_id]["is_symmetric"] for obj_id in obj_ids],
                device=points.device)
        is_not_sym = torch.logical_not(is_sym)
        dists_add_maybe_s = torch.empty_like(dists_adds)
        dists_add_maybe_s[is_sym] = dists_adds[is_sym]
        dists_add_maybe_s[is_not_sym] = dists_add[is_not_sym]
        mean_norm_add = dists_add.mean(-1)
        mean_norm_adds = dists_adds.mean(-1)
        mean_norm_add_maybe_s = dists_add_maybe_s.mean(-1)
            
        self.add_meter.update(obj_ids, mean_norm_add.cpu().numpy().tolist())
        self.adds_meter.update(obj_ids, mean_norm_adds.cpu().numpy().tolist())
        self.add_maybe_s_meter.update(obj_ids, mean_norm_add_maybe_s.cpu().numpy().tolist())

    # update for GT objects that were not detected TODO make one update funct
    def update_no_det(self, obj_ids):
        inf_norm = [np.inf for obj_id in obj_ids]
        self.add_meter.update(obj_ids, inf_norm)
        self.adds_meter.update(obj_ids, inf_norm)
        self.add_maybe_s_meter.update(obj_ids, inf_norm)


    def result(self):
        return {
            "AUC of ADD": self.add_meter.average(),
            "AUC of ADD-S": self.adds_meter.average(),
            "AUC of ADD(-S)": self.add_maybe_s_meter.average(),
        }
    
    # Print the scores per object
    def pprint_objs_str(self, gt_obj_map):
        '''
        gt_obj_map: Dict containing all objects. 
                    Map from obj_id to the preferred name for printing.
        '''
        def pad(s, w=22):
            s = str(s)
            assert len(s) <= w, f"String {s} is too long for width ({w})"
            return s + ' ' * (w-len(s))
        
        ret = ""
        result = self.result()
        ret += pad("") + '& '
        keys = ["AUC of ADD", "AUC of ADD-S"]
        for i, k in enumerate(keys):
            ret += pad(k, 15) + ('' if i==len(keys)-1 else '& ')
        ret += '\\\\\n'
        for obj_id in sorted(list(gt_obj_map.keys())):
            ret += pad(gt_obj_map[obj_id]) + '& '
            for i, k in enumerate(keys):
                _, auc_per_obj = result[k]
                ret += pad(f"{100 * auc_per_obj.get(obj_id,0):.1f}", 15) \
                        + ('' if i==len(keys)-1 else '& ')
            ret += '\\\\\n'

        ret += pad('Mean') + '& '
        for i, k in enumerate(keys):
            avg, _ = result[k]
            ret += pad(f"{100 * avg:.1f}", 15) \
                    + ('' if i==len(keys)-1 else '& ')
        ret += '\n\n'
        ret += f'AUC of ADD(-S): {100*result["AUC of ADD(-S)"][0]:.1f}\n'
        return ret

    def pprint_objs(self, gt_obj_map):
        print('===========================================================')
        print(self.pprint_objs_str(gt_obj_map))
        print('===========================================================')
        print('\n\n')

    def pprint(self):
        result = self.result()
        for k, v in result.items():
            print(f"{k}: {v[0]}")
    
    def __to_tensor(self, x, device):
        if type(x) in [np.ndarray, list, int, float]:
            return torch.tensor(x, device=device)
        elif type(x) == torch.Tensor:
            return x.to(device)
        else:
            raise ValueError()

    '''
    pred_points, gt_points: [batch_size num_pts 3] Model points transformed into camera
                            frame with predicted pose and gt pose, respectively.
    '''
    def __dists_add(self, pred_points, gt_points):
        dists = gt_points - pred_points
        return torch.norm(dists, dim=-1, p=2)

    '''
    pred_points, gt_points: [batch_size num_pts 3] Model points transformed into camera
                            frame with predicted pose and gt pose, respectively.
    '''
    def __dists_add_sym(self, pred_points, gt_points):
        return torch.norm(gt_points.unsqueeze(2)-pred_points.unsqueeze(1),dim=-1,p=2).min(2).values


if __name__ == '__main__':
    meter = AverageMeter()
    s = 0
    n = 0
    for i in range(1000):
        k = np.random.randint(1,100)
        x = np.random.normal()
        meter.update(x, k)
        s += x
        n += k
    
    print(f"Summed={s/n}, metered={meter.average()}")
