from yacs.config import CfgNode as CN
import os


def update_config(cfg, args):
	cfg.defrost()
	cfg.merge_from_file(args.cfg)
	cfg.freeze()
	return cfg

cfg = CN()

cfg.MODEL=CN()
cfg.MODEL.SPARSE_INST=CN()
cfg.MODEL.SPARSE_INST.ENCODER=CN()
cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS=256
cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES=['res3','res4','res5']

cfg.MODEL.SPARSE_INST.DECODER=CN()
cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS = 100
cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 80
cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM = 128
cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR = 2
cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM = False
cfg.MODEL.SPARSE_INST.DECODER.GROUPS = 4

cfg.MODEL.SPARSE_INST.DECODER.INST=CN()
cfg.MODEL.SPARSE_INST.DECODER.INST.DIM = 256
cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS = 4

cfg.MODEL.SPARSE_INST.DECODER.MASK=CN()
cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM = 256
cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS = 4

cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.005
cfg.MODEL.SPARSE_INST.MASK_THRESHOLD = 0.45
cfg.MODEL.SPARSE_INST.MAX_DETECTIONS = 100

cfg.MODEL.PIXEL_MEAN=[123.675, 116.280, 103.530]
cfg.MODEL.PIXEL_STD=[58.395, 57.120, 57.375]
