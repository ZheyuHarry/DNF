import mindspore
import os
import shutil
import cv2
import numpy as np

def load_checkpoint(config, model, optimizer, lr_scheduler, logger, epoch=None):
    resume_ckpt_path_1 = config['train']['resume']['1']
    resume_ckpt_path_model = config['train']['resume']['model']
    resume_ckpt_path_optimizer = config['train']['resume']['optimizer']
    resume_ckpt_path_lrscheduler = config['train']['resume']['lrscheduler']

    logger.info(f"==============> Resuming form {resume_ckpt_path_1}....................")
    logger.info(f"==============> Resuming form {resume_ckpt_path_model}....................")
    logger.info(f"==============> Resuming form {resume_ckpt_path_optimizer}....................")
    logger.info(f"==============> Resuming form {resume_ckpt_path_lrscheduler}....................")

    checkpoint_1 = mindspore.load_checkpoint(resume_ckpt_path_1)
    checkpoint_model = mindspore.load_checkpoint(resume_ckpt_path_model)
    checkpoint_optimizer = mindspore.load_checkpoint(resume_ckpt_path_optimizer)
    checkpoint_lrscheduler = mindspore.load_checkpoint(resume_ckpt_path_lrscheduler)

    mindspore.load_param_into_net(model , checkpoint_model)

    max_psnr = 0.0
    if not config.get('eval_mode', False):
        mindspore.load_param_into_net(optimizer, checkpoint_optimizer)
        mindspore.load_param_into_net(lr_scheduler, checkpoint_lrscheduler)
        if 'max_psnr' in checkpoint_1:
            max_psnr = checkpoint_1['max_psnr']
    if epoch is None and 'epoch' in checkpoint_1:
        config['train']['start_epoch'] = checkpoint_1['epoch']
        logger.info(f"=> loaded successfully '{resume_ckpt_path_1}' (epoch {checkpoint_1['epoch']})")
        logger.info(f"=> loaded successfully '{resume_ckpt_path_model}' (epoch {checkpoint_model['epoch']})")
        logger.info(f"=> loaded successfully '{resume_ckpt_path_optimizer}' (epoch {checkpoint_optimizer['epoch']})")
        logger.info(f"=> loaded successfully '{resume_ckpt_path_lrscheduler}' (epoch {checkpoint_lrscheduler['epoch']})")

    del checkpoint_1 , checkpoint_lrscheduler , checkpoint_optimizer , checkpoint_model
    return max_psnr


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config['train']['pretrained']}....................")
    checkpoint = mindspore.load_checkpoint(config['train']['pretrained'])
    param_not_load , _ = mindspore.load_param_into_net(model , checkpoint)
    logger.warning(f"This parameters may not be loaded: {param_not_load}")

    logger.info(f"=> loaded successfully '{config['train']['pretrained']}'")

    del checkpoint


def save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, logger, is_best=False):
    save_state = {'max_psnr': max_psnr,
                  'epoch': epoch,
                  'config': config}

    os.makedirs(os.path.join(config['output'], 'checkpoints'), exist_ok=True)

    save_path_1 = os.path.join(config['output'], 'checkpoints', 'checkpoint.ckpt')
    save_path_model = os.path.join(config['output'], 'checkpoints', 'model.ckpt')
    save_path_optimizer = os.path.join(config['output'], 'checkpoints', 'optimizer.ckpt')
    save_path_lrscheduler = os.path.join(config['output'], 'checkpoints', 'lrscheduler.ckpt')
    logger.info(f"{save_path_1} saving......")
    logger.info(f"{save_path_model} saving......")
    logger.info(f"{save_path_optimizer} saving......")
    logger.info(f"{save_path_lrscheduler} saving......")

    mindspore.save_checkpoint(save_state, save_path_1)
    mindspore.save_checkpoint(model, save_path_model)
    mindspore.save_checkpoint(optimizer, save_path_optimizer)
    mindspore.save_checkpoint(lr_scheduler, save_path_lrscheduler)

    logger.info(f"{save_path_1} saved")
    logger.info(f"{save_path_model} saved")
    logger.info(f"{save_path_optimizer} saved")
    logger.info(f"{save_path_lrscheduler} saved")

    if epoch % config['save_per_epoch'] == 0 or (config['train']['epochs'] - epoch) < 50:
        shutil.copy(save_path_1, os.path.join(config['output'], 'checkpoints', f'epoch_{epoch:04d}_1.ckpt'))
        shutil.copy(save_path_model, os.path.join(config['output'], 'checkpoints', f'epoch_{epoch:04d}_model.ckpt'))
        shutil.copy(save_path_optimizer, os.path.join(config['output'], 'checkpoints', f'epoch_{epoch:04d}_optimizer.ckpt'))
        shutil.copy(save_path_lrscheduler, os.path.join(config['output'], 'checkpoints', f'epoch_{epoch:04d}_lrscheduler.ckpt'))
        logger.info(f"{save_path_1} copied to epoch_{epoch:04d}_1.ckpt")
        logger.info(f"{save_path_model} copied to epoch_{epoch:04d}_model.ckpt")
        logger.info(f"{save_path_optimizer} copied to epoch_{epoch:04d}_optimizer.ckpt")
        logger.info(f"{save_path_lrscheduler} copied to epoch_{epoch:04d}_lrscheduler.ckpt")
    if is_best:
        shutil.copy(save_path_1, os.path.join(config['output'], 'checkpoints', 'model_best_1.ckpt'))
        shutil.copy(save_path_model, os.path.join(config['output'], 'checkpoints', 'model_best_model.ckpt'))
        shutil.copy(save_path_optimizer, os.path.join(config['output'], 'checkpoints', 'model_best_optimizer.ckpt'))
        shutil.copy(save_path_lrscheduler, os.path.join(config['output'], 'checkpoints', 'model_best_lrscheduler.ckpt'))
        logger.info(f"{save_path_1} copied to model_best_1.ckpt")
        logger.info(f"{save_path_model} copied to model_best_model.ckpt")
        logger.info(f"{save_path_optimizer} copied to model_best_optimizer.ckpt")
        logger.info(f"{save_path_lrscheduler} copied to model_best_lrscheduler.ckpt")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, mindspore.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def save_image_torch(img, file_path, range_255_float=True, params=None, auto_mkdir=True):
    """Write image to file.
    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)

    assert len(img.size()) == 3
    img = img.clone().cpu().detach().numpy().transpose(1, 2, 0)

    if range_255_float:
        # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
        img = img.clip(0, 255).round()
        img = img.astype(np.uint8)
    else:
        img = img.clip(0, 1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')