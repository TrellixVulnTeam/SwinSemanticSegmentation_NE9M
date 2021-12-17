import neptune.new as neptune
import json

with open('C:/Users/David/OneDrive/Phd/Results/ViT_base/20211208_152028.log.json', 'r') as file:
    results = file.readlines()
    results = [line.rstrip() for line in results]

run = neptune.init(
    project='eiphodos/Asparagus'
)

iters_per_epoch = 900
for r in results[1:]:
    stats = eval(r)
    iter = stats['iter']
    if stats['mode'] == 'train':
        if not iter % iters_per_epoch:
            epoch = int(iter / iters_per_epoch)
            run['train/decode_loss_seg'].log(stats['decode.loss_seg'], epoch)
            run['train/decode_acc_seg'].log(stats['decode.acc_seg'], epoch)
            run['train/aux_loss_seg'].log(stats['aux.loss_seg'], epoch)
            run['train/aux_acc_seg'].log(stats['aux.acc_seg'], epoch)
            run['train/loss'].log(stats['loss'], epoch)
    if stats['mode'] == 'val':
        epoch = int(iter / iters_per_epoch)
        run['val/mIoU'].log(stats['mIoU'], epoch)
        run['val/mDice'].log(stats['mDice'], epoch)
        run['val/mAcc'].log(stats['mAcc'], epoch)
        run['val/aAcc'].log(stats['aAcc'], epoch)

