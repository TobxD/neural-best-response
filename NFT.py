import torch, nfn
from torch import nn
from torch.utils.data import default_collate
from nfn.common import state_dict_to_tensors, WeightSpaceFeatures, network_spec_from_wsfeat
from nfn.layers import NPLinear, HNPPool, TupleOp

def nft_weights(models):
    """
    models should be batch of pytorch models
    """
    state_dicts = [m.state_dict() for m in models]
    wts_and_bs = [state_dict_to_tensors(sd) for sd in state_dicts]
    # Collate batch. Can be done automatically by DataLoader.
    wts_and_bs = default_collate(wts_and_bs)
    wsfeat = WeightSpaceFeatures(*wts_and_bs)

    out = nfn(wsfeat)  # NFN can now ingest WeightSpaceFeatures

    return out