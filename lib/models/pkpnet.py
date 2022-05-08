
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..labeling import kp_config
from .hg import HourglassNet
        
# Softmax over 2D feature maps (basically one per channel)
# Args
# raw: (torch.float32) [B K H W] Raw tensor output of network
def spatial_softmax(raw):
    num_kp, vh, vw  = raw.shape[1], raw.shape[2], raw.shape[3]
    prob = raw.reshape([-1, num_kp, vh*vw])
    prob = F.softmax(prob, dim=-1)
    return prob.reshape(raw.shape)

def mesh_grid(h, w, device=torch.device('cpu')):
    """
    Creates a mesh grid with normalized pixel values.
    """
    assert h == w, "Only support square images for now"
    r = torch.arange(0.5, h, 1) / (h / 2) - 1
    xx, yy = torch.meshgrid(r, -r)
    return xx.to(torch.float32).to(device), yy.to(torch.float32).to(device)

def post_process_kp(prob, z=None, calc_sigma=False):
    """
    Calculates the expected value of uv and z.

    Args:
        prob, z: output of keypoint network a image
                (batch_size, num_kp, h, w)
    Returns:
        uv: expected value of u,v location in image coordinates system
        z: expected value of z 
    """


    # NOTE: Assume prob already softmaxed
    num_kp, vh, vw = prob.shape[1], prob.shape[2], prob.shape[3]

    xx, yy = mesh_grid(vh, vw, prob.device) # each of xx, yy is (vh, vw)
    ret = {}
    sx = torch.sum(prob*xx, [2,3]) # (-1, nkp)
    sy = torch.sum(prob*yy, [2,3])
    mean_uv = torch.stack([sx,sy], -1) # (-1, nkp, 2)
    ret["uv"] = mean_uv
    if calc_sigma:
        # Get the covariance matrix of the distributions.
        # res is (-1, num_kp, vh, vw, 2)
        res = torch.stack([xx, yy], -1)[None,None,...] - mean_uv.reshape([-1,num_kp,1,1,2])
        # Matmul to 2x2 cov matrix and reduce over spatial dimensions
        # result is (-1, num_kp, 2, 2). A 2x2 covariance matrix for each keypoint.
        ret["cov"] = torch.sum(prob[...,None,None] 
                * torch.matmul(res[...,None], torch.unsqueeze(res,4)), [2,3])

    if z is not None:
        z = torch.sum(prob*z, [2,3])
        ret["z"] = torch.reshape(z, [-1, num_kp, 1])
    
    return ret

class PkpNet(nn.Module):

    def __init__(self, input_res=(256,256), calc_cov=True):
        super(PkpNet, self).__init__()
        self.input_res = input_res
        self.calc_cov = calc_cov
        self.num_kp = kp_config.num_kp()
        self.backbone = HourglassNet(nInChan=3+self.num_kp, numOutput=self.num_kp)
        # For predicting keypoint inlier mask
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(self.num_kp, self.num_kp),
        )

    def forward(self, images, boxes, prior_kp=None):
        """
        :param images: [B C H W] tensor of input images
        :param boxes: in any format accepted by torchvision.ops.roi_align
        :param prior_kp: List of Tensor[L,num_kp,*input_res] of the optional prior
                         keypoint input made with utils.make_prior_kp_input, and where
                         L is the number of bounding boxes in the corresponding
                         list index of boxes
        """
            
        # [B C H W] -> [K C H W] for K total boxes
        assert type(boxes) == list and len(boxes) == images.shape[0]
        #print("BEFORE", images.shape, len(boxes))
        images = torchvision.ops.roi_align(images, boxes, output_size=self.input_res)
        #print("AFTER", images.shape)
        if prior_kp is None:
            prior_kp = torch.zeros((images.shape[0],self.num_kp,images.shape[2],images.shape[3]),
                                    device=images.device)
        else:
            prior_kp = torch.cat(prior_kp)
        #print("PRIOR", prior_kp.shape)
        images = torch.cat([images,prior_kp], axis=1)
        
        raw = self.backbone(images) 

        # Get uv prediction and covariance
        prob = spatial_softmax(raw)
        ret = post_process_kp(prob, z=None, calc_sigma=self.calc_cov)

        # Add the raw and prob grid in case you need it
        ret["prob_logits"] = raw 
        ret["prob"] = prob 

        # Estimate the keypoint inlier mask from the avg pooled
        # raw prob grid. Return raw one so you can use bce loss with logits
        # if you want for stability.
        ret["kp_mask_logits"] = self.classifier(raw.mean(3).mean(2))
        # Sigmoid to get to [0,1] probability
        ret["kp_mask"] = torch.sigmoid(ret["kp_mask_logits"])
        return ret
        
