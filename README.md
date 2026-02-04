# 4DR-MR

This is the documentation for how to use our 4DR-MR frameworks with K-Radar dataset. We tested the frameworks on the following environment:

* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.11.0+cu113
* CUDA 11.3
* opencv 4.2.0.32
* open3d 0.15.2

paper: <a href="https://ieeexplore.ieee.org/abstract/document/11355718"> link </a>

## Requirements

1. Clone the repository
```
git clone https://github.com/kaist-avelab/K-Radar.git
cd K-Radar
```

2. Create a conda environment
```
conda create -n kradar python=3.8.13 -y
conda activate kradar
```

3. Install PyTorch (We recommend pytorch 1.11.0.)

4. Install the dependencies
```
pip install -r requirements.txt
```

5. Build packages for Rotated IoU
```
cd utils/Rotated_IoU/cuda_op
python setup.py install
```

6. Modify the code in packages
```
Add line 11: 'from .nms import rboxes' for __init__.py of nms module.
Add line 39: 'rrect = tuple(rrect)' and comment line 41: 'print(r)' in nms.py of nms module.
```

7. Build packages for OpenPCDet operations
```
cd ../../../ops
python setup.py develop
```

8. Unzip 'kradar_revised_label_v2_0.zip' in the 'tools/revise_label' directory (For the updated labeling format, please refer to [the dataset documentation](/docs/dataset.md).)

We use the operations from <a href="https://github.com/open-mmlab/OpenPCDet">OpenPCDet</a> repository and acknowledge that all code in `ops` directory is sourced from there.
To align with our project requirements, we have made several modifications to the original code and have uploaded the revised versions to our repository.
We extend our gratitude to MMLab for their great work.

## Datasets
We use various preprocessed 4D Radar pointcloud data from RadarTensor.
You can download the data RadarTensor/from_rdr_polar_3d and from_rdr_cube_xyz.

<a href="https://drive.google.com/drive/folders/1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_?usp=share_link"> Download link </a>

## Train & Evaluation
* To train the model, prepare the total dataset and run
```
python main_train_CL.py
```

* To evaluate the model, modify the path and run
```
python main_cond_C_L.py (for conditional evaluation)
```


## Model Zoo

TODO: 

(1) ${AP_{3D}}$
| Network      | 4DR-MR| RTNH   | RTNH+    | RadarPillarNet  | RPFA | SMURF | DADAN | MVFAN |
|:------------:|:-----:|:------:|:--------:|:----:|:----:|:-----:|:---------:|:---------:|
| Sedan        | 44.16 | 36.84  | 37.36    | 37.58 | 37.49 | 37.57  | 37.62    | 43.91   |
| Bus or Truck | 25.62 | 19.55  | 19.90    | 30.68 | 23.19 | 27.80  | 33.47    | 29.28   |

(2) ${AP_{BEV}}$
| Network      | 4DR-MR| RTNH   | RTNH+    | RadarPillarNet  | RPFA | SMURF | DADAN | MVFAN |
|:------------:|:-----:|:------:|:--------:|:----:|:----:|:-----:|:---------:|:---------:|
| Sedan        | 52.59  | 43.10   | 45.79  | 46.02 | 40.0 | 40.01  | 45.59   | 47.34     |
| Bus or Truck | 29.02  | 25.60   | 27.51  | 41.84 | 32.50| 39.84  | 38.36   | 38.98     |

## Citation

If you find this work is useful for your research, please consider citing:
```
@article{song2026enhanced,
  title={Enhanced 3D Object Detection via Diverse Feature Representations of 4D Radar Tensor},
  author={Song, Seung-Hyun and Paek, Dong-Hee and Dao, Minh-Quan and Malis, Ezio and Kong, Seung-Hyun},
  journal={IEEE Sensors Journal},
  year={2026},
  publisher={IEEE}
}
```
