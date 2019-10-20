import tensorflow as tf

hparams = tf.contrib.training.HParams(
    
    # Audio:
    num_mcep=34,
    fs=22050,
    fftl=1024,
    shiftms=5.0,
    minf0=40.0,
    maxf0=500.0,
    
    # Model:
    num_features=64,
    num_classes=5,

    # Data loader
    num_workers=2,
    crop_size=(35, 128),
    source_limit=2000,
    shuffle=True,

    # Training:
    batch_size=8,
    num_epochs = 30000,
    learning_rate_gen=2e-5,
    learning_rate_disc=1e-5,
    weight_decay=1e-5,

    # Save
    checkpoint_interval=5000,

    # Eval:
    max_iters=200,
    griffin_lim_iters=60,
    power=1.5,
)

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)