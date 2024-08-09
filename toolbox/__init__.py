from .metrics import averageMeter, runningScore
from .log import get_logger

from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger

from .losses.loss import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d, LovaszSoftmax, LDAMLoss, MscCrossEntropyLoss

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


# 加载数据集
def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900']


    if cfg['dataset'] == 'irseg':
        # 导入数据集
        from .datasets.irseg import IRSeg
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='test')
        # 根据标签划分数据集
        #return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

def get_model(cfg):

    if cfg['model_name'] == 'MDNet':
        from toolbox.models.paper4.model1  import MDNet
        return MDNet()
