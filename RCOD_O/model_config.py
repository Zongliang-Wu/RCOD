"""
RCOD_O Model Configuration Mapping
"""

def get_config(model_name=None, phase=None):
    from rcod_o_model import OSEDiff_reg, OSEDiff_gen, OSEDiff_test

    if phase == 'train':
        return OSEDiff_reg, OSEDiff_gen
    elif phase == 'test':
        return OSEDiff_test