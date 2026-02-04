'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['CUDA_VISIBLE_DEVICES']= '0'

from pipelines.pipeline_detection_vC_L_R2R import PipelineDetection_vC_L_R2R

PATH_CONFIG_STUDENT = './configs/cfg_RTNH_wide_CL_student.yml'
PATH_CONFIG_1 = './configs/cfg_RTNH_wide_CL_1.yml'
PATH_CONFIG_2 = './configs/cfg_RTNH_wide_CL_2.yml'
PATH_CONFIG_3 = './configs/cfg_RTNH_wide_CL_3.yml'


if __name__ == '__main__':
    pline = PipelineDetection_vC_L_R2R(path_cfg_student=PATH_CONFIG_STUDENT, mode='train', path_cfg_1=PATH_CONFIG_1, path_cfg_2=PATH_CONFIG_2, path_cfg_3=PATH_CONFIG_3)

    ### Save this file for checking ###
    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
    ### Save this file for checking ###

    pline.train_network()

    # conditional evaluation for last epoch
    pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False)
