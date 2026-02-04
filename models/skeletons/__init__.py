'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from .rdr_base import RadarBase
from .ldr_base import LidarBase
from .pvrcnn_pp import PVRCNNPlusPlus
from .second_net import SECONDNet
from .KRKD import KRKD
from .radar_KRKD import radar_KRKD
from .radar_KRCLKD import radar_KRCLKD
from .radar_prog_KD import radar_prog_KD

def build_skeleton(cfg):
    return __all__[cfg.MODEL.SKELETON](cfg)

def build_skeleton_KD(cfg):
    return __all__[cfg['MODEL'].SKELETON](cfg)

__all__ = {
    'RadarBase': RadarBase,
    'LidarBase': LidarBase,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'SECONDNet': SECONDNet,
    'KRKD': KRKD,
    'radar_KRKD': radar_KRKD,
    'radar_KRCLKD': radar_KRCLKD,
    'radar_prog_KD': radar_prog_KD,
}
