from utils.registry import FORWARD_REGISTRY



@FORWARD_REGISTRY.register(suffix='DNF')
def train_forward(config, model, data):
    raw = data['noisy_raw']
    raw_gt = data['clean_raw']
    rgb_gt = data['clean_rgb']

    rgb_out, raw_out = model(raw)
    ###### | output                          | label
    return {'rgb': rgb_out, 'raw': raw_out}, {'rgb': rgb_gt, 'raw': raw_gt}


@FORWARD_REGISTRY.register(suffix='DNF')
def test_forward(config, model, data):
    if not config['test'].get('cpu', False):
        raw = data['noisy_raw']
        raw_gt = data['clean_raw']
        rgb_gt = data['clean_rgb']
    else:
        raw = data['noisy_raw']
        raw_gt = data['clean_raw']
        rgb_gt = data['clean_rgb']
    img_files = data['img_file']
    lbl_files = data['lbl_file']

    rgb_out, raw_out = model(raw)

    ###### | output                          | label                         | img and label names
    return {'rgb': rgb_out, 'raw': raw_out}, {'rgb': rgb_gt, 'raw': raw_gt}, img_files, lbl_files


@FORWARD_REGISTRY.register(suffix='DNF')  # without label, for inference only
def inference(config, model, data):
    raw = data['noisy_raw']
    img_files = data['img_file']

    rgb_out, raw_out = model(raw)

    ###### | output                          | img names
    return {'rgb': rgb_out, 'raw': raw_out}, img_files



