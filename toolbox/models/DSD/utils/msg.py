import torch


class runMsg(object):
    def __init__(self):
        self.index = 0.0
        self.iou = 0.0
        self.ber = 0.0
        self.mae = 0.0

    def update(self, label_trues, label_preds):
        self.index += 1.0
        pred = (label_preds >= 0.5)
        gt = (label_trues >= 0.5)
        # iou
        self.iou += torch.sum((pred & gt)) / torch.sum((pred | gt))
        # ber
        N_p = torch.sum(gt) + 1e-20
        N_n = torch.sum(torch.logical_not(gt)) + 1e-20
        TP = torch.sum(pred & gt)
        TN = torch.sum(torch.logical_not(pred) & torch.logical_not(gt))
        self.ber += 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))
        # mae
        pred = torch.where(label_preds >= 0.5, torch.ones_like(label_preds), torch.zeros_like(label_preds))
        gt = torch.where(label_trues >= 0.5, torch.ones_like(label_trues), torch.zeros_like(label_trues))
        self.mae += torch.abs(pred - gt).mean()

    def get_scores(self):
        if self.index is not 0.0:
            iou = self.iou/self.index
            mae = self.mae/self.index
            ber = self.ber/self.index
        return (
            {
                "iou: ": iou,
                "mae: ": mae,
                "ber: ": ber,
            }
        )

    def reset(self):
        self.index = 0.0
        self.iou = 0.0
        self.ber = 0.0
        self.mae = 0.0


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
