# dataset settings
dataset_type = 'Dataset3dsam'
data_root = '/data/sdb/user/Deeplesion/'
anno_root = '/data/sdb/user/mmdet/data/'
train_pipeline = [
    dict(type='LoadTioImage'),
    dict(type='RescaleIntensity'),
    dict(type='ComputeAugParam')
]
test_pipeline = [
    dict(type='LoadTioImage'),
    dict(type='RescaleIntensity')
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_dir=data_root + 'nii/',
        index_file=anno_root + 'deeplesion_filename.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_dir=data_root + 'Images_nifti/',
        index_file=anno_root + 'dlt.csv',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_dir=data_root + 'Images_nifti/',
        index_file=anno_root + '/filename_test.csv',
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1)
