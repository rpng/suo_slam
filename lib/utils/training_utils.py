import torch
from collections import defaultdict

# DataParallel wrapper to properly handle list of bboxes
class DataParallelWrapper(torch.nn.DataParallel):
    def __init__(self, module):
        # Disable all the other parameters
        super(DataParallelWrapper, self).__init__(module)

    def forward(self, *inputs, **kwargs):
        assert len(inputs) == 0, "Only support forward with kwargs"
        new_inputs = [{} for _ in self.device_ids]
        batch_size = len(next(iter(kwargs.values())))
        batch_per_device = max(1, batch_size//len(self.device_ids))
        devices_needed = set() # May not need all devices if batch is 1
        # Account for if batch is less than number of devices with this counter
        for j, key in enumerate(kwargs):
            batch_consumed = 0
            assert len(kwargs[key]) == batch_size, "Inconsistent batch size"
            for i, device in enumerate(self.device_ids):
                assert isinstance(kwargs[key], list) or torch.is_tensor(kwargs[key]), \
                        f"Unsupported input type {type(kwargs[key])} for forward()"
                if i < len(self.device_ids)-1:
                    tensors = kwargs[key][i*batch_per_device:(i+1)*batch_per_device]
                else:
                    # Account for integer remainder of batch
                    tensors = kwargs[key][i*batch_per_device:]
                if isinstance(tensors, list):
                    new_inputs[i][key] = [x.to(device) for x in tensors]
                else:
                    new_inputs[i][key] = tensors.to(device)
                batch_consumed += len(tensors)
                devices_needed.add(device)
                if batch_consumed >= batch_size: # Greater if remainder
                    break
        new_inputs = new_inputs[:len(devices_needed)]
        nones = [[] for _ in devices_needed]
        replicas = self.replicate(self.module, devices_needed)
        outputs = self.parallel_apply(replicas, nones, new_inputs)
        return self.gather(outputs, self.output_device)

def collate_fn(batch, truncate_obj=None):
    """
    Since each image may have a different number of objects, 
    we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    :param batch: an iterable of N sets from __getitem__()
    :param truncate_obj: see args.py
    :return: a dict just like __getitem__ returns, but obj_ids, poses, bboxes, priors, 
             K_kps, kp_uvs, kp_masks, model_kps will become lists of tensors.
    """
    
    collated = defaultdict(list)
    for b in batch:
        for k, v in b.items():
            collated[k].append(v)
    
    # These tensors can actually be stacked since they are the same
    # size per batch
    stack_keys = ["img", "K"]
    for k in stack_keys:
        collated[k] = torch.stack(collated[k])

    if truncate_obj is not None:
        def num_obj(xlist):
            return sum([x.shape[0] for x in xlist])

        # TODO pick a random index to remove instead of the last one.
        # This would require tracking the same index across all the lists though.
        def rm_obj(xlist):
            # Reduce each list element's batch (num obj) size by 1 until 
            # there is a small enough number of them. Not very efficient,
            # but reduces the size of the code. This should only be used in
            # select cases anyways.
            orig_num_obj = num_obj(xlist)
            num_warn, max_warn = 0, len(xlist)
            i, removed = 0, 0
            while num_obj(xlist) > truncate_obj:
                if xlist[i].shape[0] > 1:
                    xlist[i] = xlist[i][:-1]
                    removed += 1
                else:
                    print(f"WARNING batch with {orig_num_obj} object being reduced " +
                          f"to {truncate_obj} causes image {i} to be completely removed")
                    num_warn += 1
                    if num_warn >= max_warn:
                        raise RuntimeError(f"ERROR too many warnings. Please increase truncate_obj.")
                i = (i + 1) % len(xlist)
                
        for k in collated.keys():
            # If list, then each element is a tensor for all the obj in one view.
            if type(collated[k]) == list:
                rm_obj(collated[k])
                assert len(collated[k]) <= truncate_obj
    
    '''
    print("+++++++++++++++===========================")
    for k, v in collated.items():
        if type(v) == torch.Tensor:
            print(f"Tensor {k}: {v.shape}")
        elif type(v) == list:
            print(f"ListTensor {k} (len {len(v)}):")
            for vi in v:
                print('\t', vi.shape)
        else:
            assert False
    print("+++++++++++++++===========================")
    '''

    return collated
