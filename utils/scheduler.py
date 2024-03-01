from mindspore import nn


def build_scheduler(config, n_iter_per_epoch=1):
    if config['lr_scheduler']['t_in_epochs']:
        n_iter_per_epoch = 1
    num_steps = int(config['epochs'] * n_iter_per_epoch)
    warmup_steps = int(config['warmup_epochs'] * n_iter_per_epoch)
    lr_scheduler = None
    if config['lr_scheduler']['type'] == 'cosine':
        lr_scheduler = nn.CosineDecayLR(
            min_lr=config['min_lr'],
            max_lr=config['max_lr'],
            decay_steps=int(config['lr_scheduler']['decay_epochs'] * n_iter_per_epoch),
        )
    elif config['lr_scheduler']['type'] == 'step':
        decay_steps = int(config['lr_scheduler']['decay_epochs'] * n_iter_per_epoch)
        lr_scheduler = nn.ExponentialDecayLR(
            learning_rate=config['lr'],
            decay_steps=decay_steps,
            decay_rate=config['lr_scheduler']['decay_rate'],
            is_stair=True,
        )
    else:
        raise NotImplementedError()

    return lr_scheduler
