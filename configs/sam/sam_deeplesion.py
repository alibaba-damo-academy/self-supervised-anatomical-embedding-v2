_base_ = [
    '../_base_/datasets/deeplesion_sam.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='Sam',
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
        norm_eval=False,
        zero_init_residual=False),
    neck=dict(
        type='FPN3d',
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
        pre_select_pos_number=2000,
        after_select_pos_number=100,
        pre_select_neg_number=2000,
        after_select_neg_number=500,
        positive_distance=5.,
        ignore_distance=3.,
        coarse_positive_distance=25.,
        coarse_ignore_distance=5.,
        coarse_z_thres=6.,
        coarse_pre_select_neg_number=250,
        coarse_after_select_neg_number=200,
        coarse_global_select_number=300,
        temperature=0.5
    ),
    test_cfg=dict(
        save_path='/data/sdc/user/result-dlt/',
        output_embedding=False
    ))
view1_pipline = [
    dict(type="ExtraAttrs", tag="view1"),
    dict(type="Crop"),
    # dict(type="RandomAffine3d"),
    # dict(type="RandomElasticDeformation"),
    dict(type="Resample"),
    dict(type="Crop", switch='fix'),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    dict(type="RandomBlur3d"),
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
    dict(type="RandomBlur3d"),
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
    dict(type='CropBackground'),
    dict(type='ComputeAugParam'),
    dict(
        type="MultiBranch", view1=view1_pipline, view2=view2_pipline
    ),
]
test_pipeline = [
    dict(type='LoadTestTioImage'),
    dict(type='Resample'),
    dict(type='RescaleIntensity'),
    dict(type="GenerateMetaInfo"),
    # dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img'])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=12,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=test_pipeline,
    )
)
find_unused_parameters = False

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
