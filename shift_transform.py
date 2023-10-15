import transform_layers as TL
import torch.nn as nn

def get_shift_module(args, eval=False):

    if args.shift_trans_type == 'rotation':
        shift_transform = TL.Rotation()
        K_shift = 4
    elif args.shift_trans_type == 'cutperm':
        shift_transform = TL.CutPerm()
        K_shift = 4
    else:
        shift_transform = nn.Identity()
        K_shift = 1

    return shift_transform, K_shift