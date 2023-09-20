_base_ = './sam_deeplesion.py'
view1_pipline = [
    dict(type="ExtraAttrs", tag="view1"),
    dict(type="Crop"),
    # dict(type="RandomAffine3d"),
    # dict(type="RandomElasticDeformation"),
    dict(type="Resample"),
    dict(type="Crop", switch='fix'),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    # dict(type="RandomBlur3d"),
    dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo"),
    dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img', 'meshgrid', 'valid'],
         meta_keys=(
             "filename",
             "tag",
             "crop_info"
         ), )
]
view2_pipline = [
    dict(type="ExtraAttrs", tag="view2"),
    dict(type="Crop"),
    # dict(type="RandomAffine3d"),
    # dict(type="RandomElasticDeformation"),
    dict(type="Resample"),
    dict(type="Crop", switch='fix'),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    # dict(type="RandomBlur3d"),
    dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo"),
    dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img', 'meshgrid', 'valid'],
         meta_keys=(
             "filename",
             "tag",
             "crop_info"

         ), )
]
train_pipeline = [
    dict(type='LoadTioImage'),
    # dict(type='RescaleIntensity'),
    # dict(type='Crop100'),
    dict(type='CropBackground'),
    dict(type='ComputeAugParam_sample'),
    dict(
        type="MultiBranch", view1=view1_pipline, view2=view2_pipline
    ),
]
test_pipeline = [
    dict(type='LoadTestTioImage', landmark=False),
    dict(type='Resample'),
    dict(type='RescaleIntensity'),
    dict(type="GenerateMetaInfo", is_train=False),
    # dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img'])
]
model = dict(
    train_cfg=dict(
        pre_select_pos_number=2000,
        after_select_pos_number=100,
        pre_select_neg_number=2000,
        after_select_neg_number=500,
        positive_distance=2.,
        ignore_distance=20.,
        coarse_positive_distance=25.,
        coarse_ignore_distance=5.,
        coarse_z_thres=6.,
        coarse_pre_select_neg_number=250,
        coarse_after_select_neg_number=200,
        coarse_global_select_number=1000,
        temperature=0.5
    ),
    test_cfg=dict(
        save_path='/data/sdd/user/results/result-dlt/',
        # save_path='/data/sdd/user/results/landmark_n/',
        output_embedding=True
    ))
lymphnode_data_root = '/data/sdd/user/processed_data/lymphnode/nii/'
anno_root = '/data/sdd/user/processed_data/ind_files/'
dlt_eval_data_root = '/data/sdd/user/rawdata/Deeplesion/Images_nifti/'
landmark_eval_data_root = '/data/sdd/user/rawdata/chestCT/CT_nifti_ln_instance/'
landmark_ce_eval_data_root = '/data/sdd/user/rawdata/chestCT/RTCT2CE_nifti_nnunet/'
deeplesion_data_root = '/data/sdd/user/processed_data/Deeplesion/nii/'
word_data_root = '/data/sde/user/WORD-V0.1.0/imagesTr/'
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=12,
    train=dict(
        data_dir=lymphnode_data_root,
        index_file=anno_root + 'lymphnode_filename.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        data_dir=landmark_eval_data_root,
        index_file=anno_root + 'landmark.csv',
        pipeline=test_pipeline,
    ),
    test=dict(
        # data_dir=word_data_root,
        # index_file=anno_root + 'word_filename.csv',
        data_dir=dlt_eval_data_root,
        # index_file=anno_root + 'deeplesion_dlt_jimmy.csv',
        index_file=anno_root + 'dlt.csv',
        pipeline=test_pipeline,
    ), )

fp16 = dict(loss_scale="dynamic")
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[18000,19000])
