_base_ = [
    '../_base_/datasets/deeplesion_sam.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='Sam_uniform_cross_volume_dual_head_mri_ct',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet18',
        depth=18,
        in_channels=1,
        spatial_strides=(2, 2, 2, 2),
        temporal_strides=(1, 1, 1, 2),
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        conv1_stride_s=1,
        pool1_stride_t=1,
        with_pool1=False,
        with_pool2=True,
        conv_cfg=dict(type='Conv3d'),
        inflate=((0, 0), (0, 0), (1, 1), (1, 1)),
        # norm_cfg = dict(type='GN',num_groups=32, requires_grad=True),
        norm_eval=False,
        zero_init_residual=False),
    neck=dict(
        type='FPN3d',
        start_level=0,
        end_level=3,
        in_channels=[64, 128, 256],
        out_channels=128,
        num_outs=3,
        conv_cfg=dict(type='Conv3d')),
    read_out_head=dict(
        type='FPN3d',
        end_level=1,
        in_channels=[512],
        out_channels=128,
        num_outs=1,
        conv_cfg=dict(type='Conv3d')),
    # model training and testing settings
    train_cfg=dict(
        meta_cfg=dict(
            prob=0.5,
        ),
        intra_cfg=dict(
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
            fine_temperature=0.5,
            coarse_temperature=0.5),
        inter_cfg=dict(
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
            fine_temperature=0.5,
            coarse_temperature=0.5)
    ),
    test_cfg=dict(
        save_path='/data/sdd/user/results/result-ctmri-landmark/',
        # save_path='/data/sdd/user/results/result-landmark_n/',
        # save_path='/data/sdd/user/results/result-dlt/',
        output_embedding=True
    ))

intra_view1_pipline = [
    dict(type="ExtraAttrs", tag="view1"),
    dict(type="ExtraAttrs", style="intra"),
    dict(type="Crop"),
    dict(type="RandomAffine3d"),
    # dict(type="RandomElasticDeformation"),
    dict(type="Resample"),
    dict(type="Crop", switch='fix'),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    # dict(type="RandomBlur3d"),
    dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo"),
    dict(type='DefaultFormatBundle3d', switch='v2'),
    dict(type='Collect3d',
         keys=['img'],
         meta_keys=(
             "filename",
             "tag",
             "crop_info",
             "style",
             'meshgrid',
             'valid'
         ), )
]
intra_view2_pipline = [
    dict(type="ExtraAttrs", tag="view2"),
    dict(type="ExtraAttrs", style="intra"),
    dict(type="Crop"),
    dict(type="RandomAffine3d"),
    # dict(type="RandomElasticDeformation"),
    dict(type="Resample"),
    dict(type="Crop", switch='fix'),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    # dict(type="RandomBlur3d"),
    dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo"),
    dict(type='DefaultFormatBundle3d', switch='v2'),
    dict(type='Collect3d',
         keys=['img'],
         meta_keys=(
             "filename",
             "tag",
             "crop_info",
             "style",
             'meshgrid',
             'valid'

         ), )
]
intra_train_pipeline = [
    dict(type='AddSuffix', suffix='_0000.nii.gz'),
    dict(type='LoadTioImageWithMask', with_mesh=True),
    dict(type='ComputeAugParam_masksample', patch_size=(96, 96, 48)),
    dict(
        type="MultiBranch", view1=intra_view1_pipline, view2=intra_view2_pipline
    ),
]

inter_view1_pipline = [
    dict(type="ExtraAttrs", tag="CT"),
    dict(type='AddSuffix', suffix='_0000.nii.gz'),
    dict(type='LoadTioImageWithMask', with_mesh=True),
    dict(type="ExtraAttrs", style="inter"),
    # dict(type='Resample', norm_spacing=(4., 4., 4.)),
    # dict(type="Crop", switch='fix'),
    dict(type="DynamicSpacing"),
    dict(type="Resample", norm_spacing=(2., 2., 2.), intra_volume=False, crop_artifacts=False, dynamic_spacing=True),
    dict(type="RandomAffine3d"),
    # dict(type="RandomElasticDeformation"),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    dict(type="RandomBlur3d"),
    # dict(type='RandomCrop3d', thres=0.3),
    dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo", with_mask=True),
    dict(type='DefaultFormatBundle3d', switch='v2'),
    dict(type='Collect3d',
         keys=['img'],
         meta_keys=(
             "filename",
             "tag",
             "volume_size",
             "style",
             "mask",
             "meshgrid",
         ), )
]
inter_view2_pipline = [
    dict(type="ExtraAttrs", tag="MRI"),
    dict(type='AddSuffixByProb', suffix=['_0001.nii.gz', '_0002.nii.gz'], p=[0.5, 0.5]),
    dict(type='LoadTioImageWithMask', with_mesh=True),
    dict(type="ExtraAttrs", style="inter"),
    # dict(type="RandomAffine3d"),
    dict(type="DynamicSpacing"),
    dict(type="Resample", norm_spacing=(2., 2., 2.), intra_volume=False, crop_artifacts=False, dynamic_spacing=True),
    dict(type="RandomAffine3d"),
    # dict(type="Crop", switch='fix'),
    # dict(type='Resample', norm_spacing=(4., 4., 4.)),
    # dict(type="RandomElasticDeformation"),
    dict(type='DynamicRescaleIntensity'),
    dict(type="RandomNoise3d"),
    dict(type="RandomBlur3d"),
    dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo", with_mask=True),
    dict(type='DefaultFormatBundle3d', switch='v2'),
    dict(type='Collect3d',
         keys=['img', 'mask'],
         meta_keys=(
             "filename",
             "tag",
             "volume_size",
             "style",
             "mask",
             "meshgrid",
         ), )
]
inter_train_pipeline = [
    # dict(type="stoppoint"),
    dict(
        type="MultiBranch", view1=inter_view1_pipline, view2=inter_view2_pipline
    ),
]

test_pipeline = [
    # dict(type='AddSuffix', suffix='_t1.nii.gz'),
    # dict(type='AddSuffixByProb', suffix=['_0001.nii.gz', '_0002.nii.gz'], p=[0.5, 0.5]),
    dict(type='LoadTestTioImage', landmark=True, suffix='_ct.nii.gz'),
    dict(type='Resample', norm_spacing=(2., 2., 2.), crop_artifacts=False),
    # dict(type='DynamicRescaleIntensity'),
    dict(type='RescaleIntensity'),
    dict(type="IsMRI", is_mri=False),
    dict(type="GenerateMetaInfo", is_train=False),
    # dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img', ],
         meta_keys=(
             "filename",
             "is_mri",
         ), )
]
test_ct_pipeline = [
    # dict(type='AddSuffix', suffix='_t1.nii.gz'),
    # dict(type='AddSuffixByProb', suffix=['_0001.nii.gz', '_0002.nii.gz'], p=[0.5, 0.5]),
    dict(type='LoadTestTioImage', landmark=True, suffix='_ct.nii.gz'),
    dict(type='Resample', norm_spacing=(2, 2, 2), crop_artifacts=False),
    # dict(type='DynamicRescaleIntensity'),
    dict(type='RescaleIntensity'),
    dict(type="IsMRI", is_mri=False),
    dict(type="GenerateMetaInfo", is_train=False),
    # dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img', ],
         meta_keys=(
             "filename",
             "is_mri",
         ), )
]
test_mri_pipeline = [
    # dict(type='AddSuffix', suffix='_t1.nii.gz'),
    # dict(type='AddSuffixByProb', suffix=['_0001.nii.gz', '_0002.nii.gz'], p=[0.5, 0.5]),
    dict(type='LoadTestTioImage', landmark=True, suffix='_t1.nii.gz'),
    dict(type='Resample', norm_spacing=(2, 2, 2), crop_artifacts=False),
    dict(type='DynamicRescaleIntensity'),
    # dict(type='RescaleIntensity'),
    dict(type="IsMRI", is_mri=True),
    dict(type="GenerateMetaInfo", is_train=False),
    # dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img', ],
         meta_keys=(
             "filename",
             "is_mri",
         ), )
]
chest_inter_data_root = '/data/sdd/user/processed_data/chestct/nii-ce-resample/'
chest_inter_mask_root = '/data/sdd/user/processed_data/chestct/nii-ce-resample-mask/'
lymphnode_intra_data_root = '/data/sdd/user/processed_data/lymphnode/nii/'

anno_root = '/data/sdd/user/processed_data/ind_files/'
dlt_eval_data_root = '/data/sdd/user/rawdata/Deeplesion/Images_nifti/'
landmark_n_eval_data_root = '/data/sdd/user/rawdata/chestCT/CT_nifti_ln_instance/'
deeplesion_data_root = '/data/sdd/user/processed_data/Deeplesion/nii/'
landmark_ce_eval_data_root = '/data/sdd/user/rawdata/chestCT/RTCT2CE_nifti_nnunet/'

flare_inter_data_root = '/data/sdd/user/processed_data/flare/nii-resample'
flare_inter_mask_root = '/data/sdd/user/processed_data/flare/nii-resample-mask/'

totolseg_inter_data_root = '/data/sdd/user/processed_data/totolsegmentaion/nii-resample'
totolseg_inter_mask_root = '/data/sdd/user/processed_data/totolsegmentaion/nii-resample-mask'

headneck_data_root = '/data/sdd/user/processed_data/head_neck_CT_MRI/image/'
headneck_mask_root = '/data/sdd/user/processed_data/head_neck_CT_MRI/mask/'

headneck_landmrk_root = '/data/sdd/user/rawdata/CT-MRI-landmark/'

word_data_root = '/data/sde/user/WORD-V0.1.0/imagesTr/'

ct_mri_landmark_ct_set = dict(
    type='Dataset3dsam',
    data_dir=headneck_landmrk_root,
    index_file=anno_root + 'ct_mri_landmark_test.csv',
    pipeline=test_ct_pipeline,
)
ct_mri_landmark_mri_set = dict(
    type='Dataset3dsam',
    data_dir=headneck_landmrk_root,
    index_file=anno_root + 'ct_mri_landmark_test.csv',
    pipeline=test_mri_pipeline,
)

lymphnode_intra_set = dict(
    type='Dataset3dsam',
    multisets=True,
    set_length=500,
    data_dir=lymphnode_intra_data_root,
    index_file=anno_root + 'lymphnode_filename.csv',
    pipeline=intra_train_pipeline,
)

headneck_intra_set = dict(
    type='Dataset3dsam',
    multisets=True,
    set_length=100,
    data_dir=headneck_data_root,
    index_file=anno_root + 'CT_MRI_filename.csv',
    pipeline=intra_train_pipeline,
    mask_dir=headneck_mask_root,
)

headneck_inter_set = dict(
    type='Dataset3dsam',
    multisets=True,
    set_length=100,
    data_dir=headneck_data_root,
    index_file=anno_root + 'CT_MRI_filename.csv',
    pipeline=inter_train_pipeline,
    mask_dir=headneck_mask_root,
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=12,
    train=dict(
        type='ConcatDataset',
        datasets=[headneck_intra_set, headneck_inter_set]
    ),
    val=dict(
        data_dir=landmark_ce_eval_data_root,
        index_file=anno_root + 'landmark_ce.csv',
        pipeline=test_pipeline,
    ),

    test=dict(
        type='ConcatDataset',
        datasets=[ct_mri_landmark_ct_set, ct_mri_landmark_mri_set]
    ), )
#     data_dir=landmark_n_eval_data_root,
#     index_file=anno_root + 'landmark.csv',
#     pipeline=test_pipeline,
# ), )
find_unused_parameters = True

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])
runner = dict(type="IterBasedRunner", max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=20)
fp16 = dict(loss_scale="dynamic")
