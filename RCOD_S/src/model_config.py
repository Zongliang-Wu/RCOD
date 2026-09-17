"""
RCOD_S Model Configuration Mapping
"""

def get_config(model_name=None, phase=None):
    if model_name in ['deg_notext_clip', 'rcod_s', 'default']:
        from rcod_s_arch.s3diff import S3Diff
        from rcod_s_arch.s3diff_tile import S3Diff as S3Diff_tile
    elif model_name in ['deg3_mlp', 'rcod_s_mem', 'mem']:
        from mem_mlp import S3Diff
        from mem_mlp import S3Diff as S3Diff_tile
    else:
        from rcod_s_arch.s3diff import S3Diff
        from rcod_s_arch.s3diff_tile import S3Diff as S3Diff_tile

    if phase == 'train':
        return S3Diff
    elif phase == 'test':
        return S3Diff_tile