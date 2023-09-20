# Training details

## Data Preparation
- The preprocessing codes are in 'misc/' directory.

### intra-volume self-supervised training dataset (original SAM)
- Take NIH Lymphnode dataset as an example:
- Execute the following command to generate resampled data:
```
# find misc/lymphnode_preprocess_crop_multi_process.py
# change the path which saves your "original NIH lymphnode .nii files", 
# add path for generated "resampled .nii files" with masks and the path of a ".csv" file contains 
# all saved .nii files name (index).
python misc/lymphnode_preprocess_crop_multi_process.py
```


### inter-volume supervised-contrastive training dataset (SAM++)
- Take TotalSegmentator dataset as an example:
- Execute the following command to generate resampled data:
```
# find misc/merge_mask_label_multi_process.py
# change the path which saves your "original TotalSegmentator dataset path", 
python misc/merge_mask_label_multi_process.py
```

### inter-volume cross-modality training dataset (Cross-SAM)
- This assume for a single patient, you have, for instance, both T1 MRI and CT images.
- Both unregistered (may have large FOV difference) pair and registered pair use the same process code.
```
python misc/mri_ct_multi_process.py
```

## How to train

- We recommend multi-gpu training.
### SAM
See [arXiv](https://arxiv.org/abs/2012.02383).
- Take NIH Lymphnode dataset as an example:
- Change the data paths in the `configs/sam/sam_NIHLN.py`

Multi-gpu training
```
bash tools/dist_train.sh configs/sam/sam_deeplesion.py $NUM_GPUS$
```
or you can use 
```
python tools/train_sam.sh configs/sam/sam_deeplesion.py  --auto-resume --no-validate
```
for single gpu training.

### SAM++

See [arXiv](https://arxiv.org/abs/2306.13988).

```
bash tools/dist_train.sh configs/samv2/samv2_NIHLN.py $NUM_GPUS$
```

### Cross-SAM

See [arXiv](https://arxiv.org/abs/2307.03535).

We use [DEEDS](https://github.com/mattiaspaul/deedsBCV) registration for deformable registration. The original DEEDS 
applyBCVfloat.cpp file can only apply the displacement field
on isotropic spacing volumes. We modified it and make it work for anisotropic spacing volumes.
The modified file is in ./DEEDS_file/applyBCVfloat_aniso.cpp.
MAKE SURE THAT THE FINEST GRID USING IN deedsBCV IS EQUAL TO THE `final_ratio` in `applyBCVfloat_aniso.cpp`.
```
cmd = '%s/deedsBCV -F %s/fixed_resample_sam_deeds.nii.gz -M %s/affined_resample_sam_deeds.nii.gz -l 3 -G 5x4x3 -L 5x4x3 -Q 2x2x1 -O %s/MRI_SAM_resample_sam_deeds' % (
                save_path, save_path, save_path, save_path)  # The finest grid of -G is 3.
                
int final_ratio = 3;   # set the final_ratio in applyBCVfloat_aniso.cpp to 3 and make. 
```
First do aggressive-augmented SAM training (intra-volume).
```
#change sam/datasets/samplers/infinite_balanced_sampler.py line 80
# before: if len(batch_buffer) >= self.batch_size - 1:
# after: if len(batch_buffer) >= self.batch_size:
# then run
bash tools/dist_train.sh configs/samv2/samv2_ct_mri_intra.py $NUM_GPUS$
# change sam/datasets/samplers/infinite_balanced_sampler.py back to original
```
After this, you can use the trained model to align the unregistered cross-modality training data using:
```
# change the path to the checkpoint and data,and do registration
python tools/regis_sam.py  # recommand confidence-score=1.5
# generate the resampled data for next stage inter-volume training
python misc/mri_ct_multi_process.py
```
Finally, you can do cross-modality learning using:
```
bash tools/dist_train.sh configs/samv2/samv2_ct_mri_inter.py $NUM_GPUS$
```

## Evaluation

- If you want to save the embeddings of all the volume from a dataset (i.e. the DeepLesion dataset) and use them later,
you can use `test_sam.py` or `dist_test.sh`.
```
# assume you want to save the SAM++ embeddings, first find 
#.configs/samv2/samv2_NIHLN.py
#set the correct path contain the test dataset for the data config section.
#data = dict(
#    test=dict(
#        data_dir=dlt_eval_data_root,
#        index_file=anno_root + 'dlt.csv',
#        pipeline=test_pipeline,
#    ), )
#Also, set the output_embedding = False in the test_cfg section:
#    test_cfg=dict(
#        save_path='/data/sdd/user/results/result-dlt/',
#        output_embedding=False) # False: save the embeeding to the save_path; True: return the embdding
# Run
dist_tesh.sh #PATH_TO_CONFIG #NUM_GPU
# The embeddings for all cases will be saved in the save_path. This is useful for evaluations like deeplesion tracking performance.
```
-Results on the DeepLesion lesion tracking test set [(link)](https://github.com/JimmyCai91/DLT):

First save all the test case embeddings to the 'save_path', like mentioned above, and run
```
python tools/deeplesion_eval_metric_visuall_multipoints_dlt_stable.py
```
The result looks like 
```
960 928 32 # total case, number of matching within the bboxes, number of failed cases
0.9666666666666667  # ratio of matching within the bboxes, i.e. CPM@Radius we use the save method that TLT use
960 848 112 
0.8833333333333333 # CPM@10MM
5.569163221639997 5.880392643035999 # MED
2.3488916527364885 3.1877845193851555 # MED_x
2.1009860559717146 2.495776864351786# MED_y
3.777555157827975 5.000236689673672# MED_z
```
