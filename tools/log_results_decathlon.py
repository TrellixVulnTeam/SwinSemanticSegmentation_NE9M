import neptune.new as neptune
import json
import argparse

parser = argparse.ArgumentParser(description='Log to Neptune')
parser.add_argument('logfile', help='Path to mmseg json log file')
args = parser.parse_args()

with open(args.logfile, 'r') as file:
    results = file.readlines()
    results = [line.rstrip() for line in results]

run = neptune.init(
    project='eiphodos/Asparagus'
)

for r in results[1:]:
    stats = eval(r)
    iter = stats['iter']
    if not iter % 1000:
        if stats['mode'] == 'train':
            run['train/decode_loss_seg'].log(stats['decode.loss_seg'], iter)
            run['train/decode_acc_seg'].log(stats['decode.acc_seg'], iter)
            run['train/aux_loss_seg'].log(stats['aux.loss_seg'], iter)
            run['train/aux_acc_seg'].log(stats['aux.acc_seg'], iter)
            run['train/loss'].log(stats['loss'], iter)
        if stats['mode'] == 'val':
            run['val/mIoU'].log(stats['mIoU'], iter)
            run['val/mDice'].log(stats['mDice'], iter)
            run['val/mAcc'].log(stats['mAcc'], iter)
            run['val/aAcc'].log(stats['aAcc'], iter)
            run['val/liverIoU'].log(stats['liverIoU'], iter)
            run['val/liverDice'].log(stats['liverDice'], iter)
            run['val/liverAcc'].log(stats['liverAcc'], iter)
            run['val/cancerIoU'].log(stats['cancerIoU'], iter)
            run['val/cancerDice'].log(stats['cancerDice'], iter)
            run['val/cancerAcc'].log(stats['cancerAcc'], iter)

