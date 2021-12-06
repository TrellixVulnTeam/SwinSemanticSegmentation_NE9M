import neptune.new as neptune
import json

with open('/home/david/SemSegResults/Swin_Random/20211126_172459.log.json', 'r') as file:
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
            prev_epoch = epoch
            run['train/decode_loss_seg'].log(stats['decode.loss_seg'], epoch)
            run['train/decode_acc_seg'].log(stats['decode.acc_seg'], epoch)
            run['train/aux_loss_seg'].log(stats['aux.loss_seg'], epoch)
            run['train/aux_acc_seg'].log(stats['aux.acc_seg'], epoch)
            run['train/loss'].log(stats['loss'], epoch)
            run['train/lr'].log(stats['lr'], epoch)
    if stats['mode'] == 'val':
        run['val/mIoU'].log(stats['mIoU'], epoch)
        run['val/mDice'].log(stats['mDice'], epoch)
        run['val/mAcc'].log(stats['mAcc'], epoch)
        run['val/aAcc'].log(stats['aAcc'], epoch)

