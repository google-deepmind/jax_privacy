# Image Classification Experiments


## Intro


- An experiment can be run by executing from this directory:

```
python experiment.py <relative/path/to/config.py>
```

where the config file contains all relevant hyper-parameters for the experiment.

- The main config hyper-parameters to update per experiment are:

  - Augmult: `config.experiment_kwargs.config.data.augmult`
  - Batch-size: `config.experiment_kwargs.config.training.batch_size.init_value`
  - Learning-rate value: `config.experiment_kwargs.config.optimizer.lr.init_value`
  - Model definition: `config.experiment_kwargs.config.model`
  - Noise multiplier sigma: `config.experiment_kwargs.config.training.dp.noise.std_relative`
  - Number of updates: `config.experiment_kwargs.config.num_updates`
  - Privacy budget (delta): `config.experiment_kwargs.config.dp.target_delta`
  - Privacy budget (epsilon): `config.experiment_kwargs.config.dp.stop_training_at_epsilon`

Note: we provide examples of configurations for various experiments. To
reproduce the results of our paper, please refer to the hyper-parameters listed
 in Appendix C and update the relevant configuration file.

## Training from Scratch on CIFAR-10

```
python run_experiment.py --config=configs/cifar10_wrn_16_4_eps1.py
```


## Training from Scratch on ImageNet

```
python run_experiment.py --config=configs/imagenet_nf_resnet_50_eps8.py
```

## Fine-tuning on CIFAR

```
python run_experiment.py --config=configs/cifar100_wrn_28_10_eps1_finetune.py
```

See `jax_privacy/src/training/image_classsification/config_base.py` for the available pre-trained models.

## Additional Details

- The number of updates given in the config is ignored if `stop_training_at_epsilon` is specified, in which case the training automatically stops when the total privacy budget has been spent.
- The `auto_tune` feature in the config can be used to, for example, calibrate the noise multiplier under a pre-specified privacy budget, number of iterations and batch-size.
- The code supports using an `optax` learning-rate schedule, though we do not use it in practice to get our best results.
- The use of a batch-size schedule is not guaranteed to work at this time.
