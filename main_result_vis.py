'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import torch
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pipelines.pipeline_detection_vC_L_R2R import PipelineDetection_vC_L_R2R
from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

CONFIDENCE_THR = 0.7
    
x_min, y_min, z_min, x_max, y_max, z_max = [0.,-16.,-2.,72.,16.,7.6]

rdr_path = '/media/ave/d72a86b4-824b-4acc-8b10-4c73885e8fe7/kradar-dataset/pc01p/9'
ldr_path = '/media/ave/d72a86b4-824b-4acc-8b10-4c73885e8fe7/kradar-dataset/radar_bin_lidar_bag_files/generated_files/9'
label_path = 'tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL/9'
plt_save_path = '/home/ave/song/vis_results/4DR-MR_vis/9/rtnh_result'
plt_hm_save_path = '/home/ave/song/vis_results/4DR-MR_vis/9/rtnh_hm'

seq = 'seq_9_2_00648_00625'
# seq = 'seq_18_1_00184_00149'
# seq = 'seq_52_2_00152_00150'
prefix = 'seq_9_2 _'
num1, num2 = map(int, seq[len(prefix):].split('_'))

is_plt_label = True


def generate_random_color():
    return np.random.rand(3,)

# 객체 ID별로 랜덤 색상을 할당합니다. 실제 사용 시에는 필요한 만큼의 ID를 할당해야 합니다.
object_colors = {object_id: generate_random_color() for object_id in range(0, 1000)}

def ensure_directory(file_path):
    ### ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
    PATH_CONFIG_STUDENT = './configs/cfg_RTNH_wide_CL_student.yml'
    PATH_CONFIG_1 = './configs/cfg_RTNH_wide_CL_1.yml'
    PATH_CONFIG_2 = './configs/cfg_RTNH_wide_CL_2.yml'
    PATH_CONFIG_3 = './configs/cfg_RTNH_wide_CL_3.yml'
    # PATH_MODEL = './pretrained/4DR-MR_10.pt' 
    PATH_MODEL = './pretrained/rtnh_01p_29.pt'

    # pline = PipelineDetection_vC_L_R2R(path_cfg_student=PATH_CONFIG_STUDENT, mode='test', path_cfg_1=PATH_CONFIG_1, path_cfg_2=PATH_CONFIG_2, path_cfg_3=PATH_CONFIG_3)
    # pline.load_dict_model(pline.model.model_student, PATH_MODEL)
    # pline.model.model_student.eval()

    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='test')
    pline.load_dict_model(PATH_MODEL)
    pline.network.eval()

    dataset_loaded = pline.dataset_test
    data_loader = torch.utils.data.DataLoader(dataset_loaded,
            batch_size = 1, shuffle = False,
            collate_fn = pline.dataset_test.collate_fn,
            num_workers = pline.cfg.OPTIMIZER.NUM_WORKERS)
    
    for dict_item in tqdm(data_loader):
        sq = dict_item['meta'][0]['seq']
        # if sq == '11':
        #     break
        # if sq != '10':
        #     continue
        
        
        new_data = f"{prefix}{num1:05d}_{num2:05d}"
        rdr_data = osp.join(rdr_path,'rpc_' + new_data.split('_')[3] + '.npy')
        ldr_data = osp.join(ldr_path,'os2-64', 'os2-64_' + new_data.split('_')[4] + '.pcd')
        label_data = osp.join(label_path, new_data.split('_')[3] + '_' + new_data.split('_')[4] + '.txt')

        # load once and keep as torch tensor for model forward
        rdr_sparse_np = np.load(rdr_data).reshape(-1, 11)
        dict_item['rdr_sparse'] = torch.from_numpy(rdr_sparse_np).float()
        # dict_item = pline.model.model_student(dict_item)
        dict_item = pline.network(dict_item)

        dataset = pline.dataset_test
        points = dataset.get_ldr64_from_path(ldr_data)
        points = points[np.where(
            (points[:, 0] > x_min) & (points[:, 0] < x_max) &
            (points[:, 1] > y_min) & (points[:, 1] < y_max) &
            (points[:, 2] > z_min) & (points[:, 2] < z_max))]
        x_points = points[:, 0]
        y_points = points[:, 1]

        
        pc_radar = rdr_sparse_np[:, :4]
        points_rdr = pc_radar[np.where(
        (pc_radar[:, 0] > x_min) & (pc_radar[:, 0] < x_max) &
        (pc_radar[:, 1] > y_min) & (pc_radar[:, 1] < y_max) &
        (pc_radar[:, 2] > z_min) & (pc_radar[:, 2] < z_max))]
        rdr_x_points = points_rdr[:, 0]
        rdr_y_points = points_rdr[:, 1]
        rdr_intensity = points_rdr[:, 3]+0.1
        
        #label data
        label = []
        with open(label_data) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                else:
                    label.append(line)

         #plot data
        plt.figure(figsize=(5, 10))
        scatter_rdr = plt.scatter(-rdr_y_points, rdr_x_points, s=2, c=rdr_intensity, cmap='viridis_r', vmin=0.1, vmax=0.3, alpha=0.7)
        scatter = plt.scatter(-y_points, x_points, s=0.3, c='black', cmap='viridis_r', vmin=0, vmax=20, alpha=0.05)

        #label visualize
        if is_plt_label == True:
            for obj in label:
                obj = list(map(str.strip, obj.split(',')))
                _, cls_id, cls_name, x, y, z, th, l, w, h = obj
                # if cls_name != 'Sedan':
                #     continue
                x, y, th, l, w, h = float(x)-2.54, float(y)+0.3, float(th)*np.pi/180, float(l), float(w), float(h)
                y = -y
                if x > x_max or y > y_max or y < y_min:
                    continue
                corners = np.array([[-l, -w], [l, -w], [l, w], [-l, w]])
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                rotated_corners = np.dot(corners, R.T) + np.array([x, y])
                for k in range(4):
                    l = (k + 1) % 4
                    plt.plot([rotated_corners[k][1], rotated_corners[l][1]], 
                            [rotated_corners[k][0], rotated_corners[l][0]], 
                            color=[0,0,0],linewidth=2)
                text_x, text_y = rotated_corners[1][1]+0.5, rotated_corners[1][0]+w/2
                plt.text(text_x, text_y, cls_id, color=[0,0,0],fontsize = 12)


        pred_dicts = dict_item['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

        for idx_pred in range(len(pred_labels)):
            x, y, z, l, w, h, th = pred_boxes[idx_pred]
            score = pred_scores[idx_pred]
            y = -y
            l, w, h = l/2, w/2, h/2
            if x > x_max or y > y_max or y < y_min:
                    continue
            if score > CONFIDENCE_THR:
                corners = np.array([[-l, -w], [l, -w], [l, w], [-l, w]])
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                rotated_corners = np.dot(corners, R.T) + np.array([x, y])
                for k in range(4):
                    l = (k + 1) % 4
                    plt.plot([rotated_corners[k][1], rotated_corners[l][1]], 
                            [rotated_corners[k][0], rotated_corners[l][0]], 
                            color=[1,0,0],linewidth=2)
            else:
                continue
        
        ax = plt.gca()
        ax.set_autoscale_on(False)
        ax.set_xlim(-y_max, -y_min)
        ax.set_ylim(x_min, x_max)
        ax.set_aspect('equal', adjustable='box')
        
        plt_file = osp.join(plt_save_path, str(num1) + '.png')
        ensure_directory(plt_file)
        plt.savefig(plt_file,format='png',dpi=300)
        plt.close()
        # plt.show()

        # plt.figure(figsize=(5, 10))
        # feature_map = dict_item['bev_feat'].detach().cpu().numpy() #'bev_layer': KD, 'bev_feat': vanila
        # hm = feature_map[0][0:11].mean(axis=0)[:-1]
        # plt.imshow(hm, cmap='hot', interpolation='nearest')

        # plt_hm_file = osp.join(plt_hm_save_path, str(num1) + '.png')
        # ensure_directory(plt_hm_file)
        # plt.savefig(plt_hm_file,format='png',dpi=300)
        # plt.close()

        num1 += 1
        num2 += 1
        
        if num1 > 855:
            break
