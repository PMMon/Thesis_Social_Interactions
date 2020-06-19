import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from torch.utils.data import DataLoader
from data.trajectories import TrajectoryDataset, seq_collate
from data.trajectories_scene import TrajectoryDatasetEval, seq_collate_eval

# ================================= Description =================================
# Script defines DataLoader. The Trajectories can either be loaded sequence-wise
# (seq_collate_eval & TrajectoryDatasetEval) or they are processed one after another/
# trajectory-wise (seq_collate & TrajectoryDataset)
# ===============================================================================

def data_loader(args, config, phase, device = 'cpu', logger = None):
    """
    Loads data corresponding to model_type and phase (train, val, test)
    """
    if args.model_type == "social-lstm" or args.model_type == "lstm":
        collate_fn = seq_collate_eval
        dataset_fn = TrajectoryDatasetEval
    else:
        collate_fn = seq_collate
        dataset_fn = TrajectoryDataset

    if phase == "test":
        data_augmentation = args.data_augmentation
        shuffle = False
        batch_size = 128
        skip = args.obs_len+args.pred_len
    elif phase == "train":
        data_augmentation = args.data_augmentation
        shuffle = True
        batch_size = args.batch_size
        skip = args.skip
    elif phase == "val":
        data_augmentation = args.data_augmentation
        shuffle = True
        batch_size = args.batch_size
        skip = args.obs_len+args.pred_len
    else: 
        raise AssertionError('"phase" must be either train, val or test.')


    dset = dataset_fn(
        dataset_name = args.dataset_name,
        phase = phase,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        data_augmentation = data_augmentation,
        skip=skip,
        max_num= args.max_num,
        load_occupancy= config.load_occupancy,
        logger = logger,
        padding=args.padding,
        dataset_type = args.dataset_type,
        analyse_real_dataset = args.analyse_real_dataset
        )

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
        )

    logger.info("Loading %s set: %s samples" % (phase, dset.__len__()))

    return dset, loader


