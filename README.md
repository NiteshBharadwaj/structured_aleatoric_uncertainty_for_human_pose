# Structured Aleatoric Uncertainty In Human Pose Estimation

This provides codebase for the [CVPR 2019 Workshop Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Gundavarapu_Structured_Aleatoric_Uncertainty_in_Human_Pose_Estimation_CVPRW_2019_paper.pdf)

Note/TODO: Currently, only the evaluation code for pre-trained models and some skeleton code is provided. Yet to complete end-end training pipeline.
This codebase and Readme.md build upon [Integral Human Pose Regression](https://github.com/JimmySuen/integral-human-pose) codebase.

[Loss Function](pytorch_projects/common_pytorch/common_loss/weighted_mse.py
[Network](pytorch_projects/common_pytorch/blocks/resnet_direct_regression.py)


## Preparation for Training & Testing
1. Download MPII image from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)
2. Organize data like this
```
${PROJECT_ROOT}
 `-- data
     `-- mpii
        |-- images
        |-- annot
        |-- mpii_train_cache
        |-- mpii_valid_cache
     `-- hm36
        |-- images
        |-- annot
        |-- HM36_train_cache
        |-- HM36_validmin_cache
```

## Usage

### Test
To run evaluations on MPII Val dataset

Place the [models](https://drive.google.com/drive/folders/1HFTbwz3o0-6dPvS6wdjS67L2zJeDSKVF) in [pytorch_projects/integral_human_pose/output/](pytorch_projects/integral_human_pose/output/)
```bash
cd pytorch_projects/integral_human_pose
python3 test.py --cfg experiments/hm36/resnet50v1_ft/d-mh_ps-256_dj_l1_adam_bs32-4gpus_x140-90-120/lr1e-3_u.yaml --dataroot ../../data/ --model output/covariance.pth.tar --is_cov True
python3 test.py --cfg experiments/hm36/resnet50v1_ft/d-mh_ps-256_dj_l1_adam_bs32-4gpus_x140-90-120/lr1e-3_u.yaml --dataroot ../../data/ --model output/diag.pth.tar --is_cov False
```
## Cite
If you find our paper useful in your research, please consider citing:
```
@article{sun2017integral,
  title={Integral human pose regression},
  author={Sun, Xiao and Xiao, Bin and Liang, Shuang and Wei, Yichen},
  journal={arXiv preprint arXiv:1711.08229},
  year={2017}
}
@article{gundavarapu2019structured,
  title={Structured Aleatoric Uncertainty in Human Pose Estimation.},
  author={Gundavarapu, Nitesh B and Srivastava, Divyansh and Mitra, Rahul and Sharma, Abhishek and Jain, Arjun},
  journal={CVPR Workshops},
  year={2019}
}
```
