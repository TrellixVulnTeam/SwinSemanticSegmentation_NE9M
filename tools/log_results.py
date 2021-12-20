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

prev_epoch = 0
for r in results[1:]:
    stats = eval(r)
    epoch = stats['epoch']
    if stats['mode'] == 'train':
        if epoch > prev_epoch:
            run['train/decode_loss_seg'].log(stats['decode.loss_seg'], epoch)
            run['train/decode_acc_seg'].log(stats['decode.acc_seg'], epoch)
            run['train/aux_loss_seg'].log(stats['aux.loss_seg'], epoch)
            run['train/aux_acc_seg'].log(stats['aux.acc_seg'], epoch)
            run['train/loss'].log(stats['loss'], epoch)
            prev_epoch = epoch
    if stats['mode'] == 'val':
        run['val/mIoU'].log(stats['mIoU'], epoch)
        run['val/mDice'].log(stats['mDice'], epoch)
        run['val/mAcc'].log(stats['mAcc'], epoch)
        run['val/aAcc'].log(stats['aAcc'], epoch)
        run['val/liverIoU'].log(stats['liverIoU'], epoch)
        run['val/liverDice'].log(stats['liverDice'], epoch)
        run['val/liverAcc'].log(stats['liverAcc'], epoch)
        run['val/cancerIoU'].log(stats['cancerIoU'], epoch)
        run['val/cancerDice'].log(stats['cancerDice'], epoch)
        run['val/cancerAcc'].log(stats['cancerAcc'], epoch)

