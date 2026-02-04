'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pipelines.pipeline_detection_vC_L_R2R import PipelineDetection_vC_L_R2R

if __name__ == '__main__':
    PATH_CONFIG_STUDENT = './configs/cfg_RTNH_wide_CL_student.yml'
    PATH_CONFIG_1 = './configs/cfg_RTNH_wide_CL_1.yml'
    PATH_CONFIG_2 = './configs/cfg_RTNH_wide_CL_2.yml'
    PATH_CONFIG_3 = './configs/cfg_RTNH_wide_CL_3.yml'
    
    # PATH_MODEL = './KD_logs/RTNH_01p_CLKD/models/model_10.pt'
    PATH_MODEL = './pretrained/4DR-MR_10.pt' 
    # PATH_MODEL = './logs/exp_250503_195231_RTNH/models/st_RTNH20_14.pt'
    # PATH_MODEL = './logs/exp_250502_175026_RTNH/models/st_cacfar_12.pt'
    # PATH_MODEL = './KD_logs/exp_240723_202833_RTNH/models/st_RTNH_wide_model_10.pt'
    # PATH_MODEL = './logs/mt_RTNH_wcfar/models/model_8.pt'

    pline = PipelineDetection_vC_L_R2R(path_cfg_student=PATH_CONFIG_STUDENT, mode='test', path_cfg_1=PATH_CONFIG_1, path_cfg_2=PATH_CONFIG_2, path_cfg_3=PATH_CONFIG_3)
    pline.load_dict_model(pline.model.model_student, PATH_MODEL)
    
    pline.model.model_student.eval()

    # Save the code for identification
    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
    
    pline.validate_kitti_conditional(list_conf_thr=[0.3, 0.5, 0.7], is_subset=False, is_print_memory=False)
