from models.resnet14 import ResNet14
from models.res18_itracker import gaze_res18_itracker
from models.res18_gsc import res18_gs_convertor
from models.res18_trans_gsc import res18_trans_gsc
from models.gazeformer import GazeFormer

def create_model(args):
    in_size = args.input_size[0]

    if args.model == 'resnet14':
        model = ResNet14(in_size)
    elif args.model == 'res18_itracker':
        model = gaze_res18_itracker()
    elif args.model == 'res18_gsc':
        model = res18_gs_convertor(with_la=args.with_la)
    elif args.model == 'trans_gsc':
        model = res18_trans_gsc(with_la=args.with_la)
    elif args.model == 'gazeformer':
        model = GazeFormer(with_la=args.with_la)
    else:
        model = None
    
    return model

