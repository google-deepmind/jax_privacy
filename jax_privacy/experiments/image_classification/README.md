# Image Classification Experiments

Reproducing experiments of the paper "Unlocking High-Accuracy Differentially Private Image Classification through Scale"

This work is available on arXiv at [this link](https://arxiv.org/abs/2204.13650).
If you use it, please cite the following [bibtex reference](https://github.com/google-deepmind/jax_privacy/blob/main/bibtex/de2022unlocking.bib).

The following instructions assume that our package has been installed through
[option 2](https://github.com/google-deepmind/jax_privacy#install-option2).

## Intro


- An experiment can be run by executing from this directory:

```
python run_experiment_loop.py --config=<relative/path/to/config.py>
```

where the config file contains all relevant hyper-parameters for the experiment.

- The main config hyper-parameters to update per experiment are:

  - Augmult: `config.experiment_kwargs.config.data.augmult`
  - Batch-size: `config.experiment_kwargs.config.training.batch_size.init_value`
  - Learning-rate value: `config.experiment_kwargs.config.optimizer.lr.kwargs.value`
  - Model definition: `config.experiment_kwargs.config.model`
  - Noise multiplier sigma: `config.experiment_kwargs.config.training.dp.noise_multiplier`
  - Number of updates: `config.experiment_kwargs.config.num_updates`
  - Privacy budget (delta): `config.experiment_kwargs.config.dp.delta`
  - Privacy budget (epsilon): `config.experiment_kwargs.config.dp.auto_tune_target_epsilon`

Note: we provide examples of configurations for various experiments. To
reproduce the results of our paper, please refer to the hyper-parameters listed
 in Appendix C and update the relevant configuration file.

## Training from Scratch on CIFAR-10

```
python run_experiment_loop.py --config=configs/cifar10_wrn_16_4_eps1.py
```


## Training from Scratch on ImageNet

```
python run_experiment_loop.py --config=configs/imagenet_nf_resnet_50_eps8.py
```

## Fine-tuning on CIFAR

```
python run_experiment_loop.py --config=configs/cifar100_wrn_28_10_eps1_finetune.py
```

See `jax_privacy/experiments/image_classification/config_base.py` for the available pre-trained models.

## Additional Details

- Training and evaluation accuracies throughout training will be printed to the console.
- If you are observing Out of Memory errors with the default configs, consider reducing the value of `config.experiment_kwargs.config.training.batch_size.per_device_per_step` to ensure the number of examples processed each time step fits in memory. This might make training slower, but will not change the effective batch-size used for each model update. Note that `config.experiment_kwargs.config.training.batch_size.init_value` should be divisible by `config.experiment_kwargs.config.training.batch_size.per_device_per_step` times the number of accelerators in your machine.
- The number of updates given in the config is ignored if `stop_training_at_epsilon` is specified, in which case the training automatically stops when the total privacy budget has been spent.
- The `auto_tune` feature in the config can be used to, for example, calibrate the noise multiplier under a pre-specified privacy budget, number of iterations and batch-size.
- The code supports using an `optax` learning-rate schedule, though we do not use it in practice to get our best results.
- The use of a batch-size schedule is not guaranteed to work at this time.
