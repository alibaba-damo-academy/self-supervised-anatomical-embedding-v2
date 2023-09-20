_base_ = [
    '../_base_/datasets/deeplesion_sam.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='Sam_uniform_cross_volume_dual_head_mean_vector',
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
    # sem_neck=dict(
    #     type='FPN3d',
    #     start_level=0,
    #     end_level=4,
    #     in_channels=[64, 128, 256,512],
    #     out_channels=128,
    #     num_outs=4,
    #     conv_cfg=dict(type='Conv3d')),
    read_out_head=dict(
        type='FPN3d',
        end_level=1,
        in_channels=[512],
        out_channels=128,
        num_outs=1,
        conv_cfg=dict(type='Conv3d')),
    superloss=dict(type='SupConMeanLoss',
                   temperature=0.1,
                   base_temperature=1.0,
                   ),
    # model training and testing settings
    train_cfg=dict(
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
            temperature=0.5),
        inter_cfg=dict(
            sampling=True,
            ignore_by_distance=False,
            sampling_pos_number=2000,
            sampling_neg_number=3000,
            positive_distance=16.,
            ignore_distance=8,
            no_overlap_pos_number=100,
            no_overlap_neg_number=200,
            temperature=0.5)
    ),
    test_cfg=dict(
        # save_path='/data/sdd/user/results/result-word/',
        # save_path='/data/sdd/user/results/result-landmark_n/',
        save_path='/data/sdd/user/results/result-dlt/',
        output_embedding=True
    ))

intra_view1_pipline = [
    dict(type="ExtraAttrs", tag="view1"),
    dict(type="ExtraAttrs", style="intra"),
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
    # dict(type="RandomAffine3d"),
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
    dict(type='LoadTioImage'),
    # dict(type='RescaleIntensity'),
    # dict(type='Crop100'),
    dict(type='CropBackground'),
    dict(type='ComputeAugParam_sample'),
    dict(
        type="MultiBranch", view1=intra_view1_pipline, view2=intra_view2_pipline
    ),
]

inter_view1_pipline = [
    dict(type="ExtraAttrs", tag="im1"),

    dict(type="SeperateInfoByImage"),
    dict(type='LoadTioImageWithMask', with_mesh=False),
    dict(type="ExtraAttrs", style="inter"),

    # dict(type='Resample', norm_spacing=(4., 4., 4.)),
    # dict(type="Crop", switch='fix'),
    dict(type="Resample", norm_spacing=(2., 2., 2.), intra_volume=False, crop_artifacts=False),
    # dict(type="RandomElasticDeformation"),
    dict(type='Cropz', x_slice=180, y_slice=180, z_slice=120),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    dict(type="RandomBlur3d"),
    # dict(type='RandomCrop3d', thres=0.3),
    # dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo", with_mask=True),
    dict(type='DefaultFormatBundle3d', switch='v2'),
    dict(type='Collect3d',
         keys=['img'],
         meta_keys=(
             "filename",
             "tag",
             "volume_size",
             "paired",
             "style",
             "mask"
         ), )
]
inter_view2_pipline = [
    dict(type="ExtraAttrs", tag="im2"),

    dict(type="SeperateInfoByImage"),
    dict(type='LoadTioImageWithMask', with_mesh=False),
    dict(type="ExtraAttrs", style="inter"),
    # dict(type="RandomAffine3d"),

    dict(type="Resample", norm_spacing=(2., 2., 2.), intra_volume=False, crop_artifacts=False),
    dict(type='Cropz', x_slice=180, y_slice=180, z_slice=120),
    # dict(type="Crop", switch='fix'),
    # dict(type='Resample', norm_spacing=(4., 4., 4.)),
    # dict(type="RandomElasticDeformation"),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    dict(type="RandomBlur3d"),
    # dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo", with_mask=True),
    dict(type='DefaultFormatBundle3d', switch='v2'),
    dict(type='Collect3d',
         keys=['img', 'mask'],
         meta_keys=(
             "filename",
             "tag",
             "volume_size",
             "paired",
             "style",
             "mask"
         ), )
]
inter_train_pipeline = [
    dict(type='GeneratePairedImagesMaskInfo'),
    dict(
        type="MultiBranch", view1=inter_view1_pipline, view2=inter_view2_pipline
    ),
]
test_pipeline = [
    dict(type='LoadTestTioImage', landmark=False),
    dict(type='Resample', norm_spacing=(2., 2., 2.)),
    dict(type='RescaleIntensity'),
    dict(type="GenerateMetaInfo", is_train=False),
    # dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img'])
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

word_data_root = '/data/sde/user/WORD-V0.1.0/imagesTr/'
lymphnode_intra_set = dict(
    type='Dataset3dsam',
    multisets=True,
    set_length=500,
    data_dir=lymphnode_intra_data_root,
    index_file=anno_root + 'lymphnode_filename.csv',
    pipeline=intra_train_pipeline,
)
chest_inter_set = dict(
    type='Dataset3dsamCrossVolume',
    multisets=True,
    set_length=500,
    data_dir=chest_inter_data_root,
    mask_dir=chest_inter_mask_root,
    index_file=anno_root + 'chest_ce_filename.csv',
    pipeline=inter_train_pipeline,
)

flare_inter_set = dict(
    type='Dataset3dsamCrossVolume',
    multisets=True,
    set_length=2000,
    data_dir=flare_inter_data_root,
    mask_dir=flare_inter_mask_root,
    index_file=anno_root + 'flare_filename.csv',
    pipeline=inter_train_pipeline,
)

totalseg_inter_set = dict(
    type='Dataset3dsamCrossVolume',
    multisets=True,
    set_length=1200,
    data_dir=totolseg_inter_data_root,
    mask_dir=totolseg_inter_mask_root,
    index_file=anno_root + 'totolseg_filename.csv',
    pipeline=inter_train_pipeline,
)

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=12,
    train=dict(
        type='ConcatDataset',
        datasets=[lymphnode_intra_set, totalseg_inter_set]
    ),
    val=dict(
        data_dir=landmark_ce_eval_data_root,
        index_file=anno_root + 'landmark_ce.csv',
        pipeline=test_pipeline,
    ),

    test=dict(
        # data_dir=word_data_root,
        # index_file=anno_root + 'word_filename.csv',
        data_dir=dlt_eval_data_root,
        index_file=anno_root + 'dlt.csv',
        pipeline=test_pipeline,
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
