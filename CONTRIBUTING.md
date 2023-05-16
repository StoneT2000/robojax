# Contributing to Robojax

Thanks for contributing to Robojax! To get started make sure to setup your development environment properly.

```
git clone https://github.com/StoneT2000/robojax
conda create -n "robojax" "python==3.9"
conda activate robojax
pip install -e .[dev]
```

Jax must be separately installed. To install jax with cuda support, follow the instructions on their [README](https://github.com/google/jax).

```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

To maintain code quality and standards, this repo uses [pre-commit](https://pre-commit.com/).

Install it and set it up by running
```
pre-commit
```

By default, this is setup for running and testing Gymnasium's default environments. Note that for some gymnasium environments there are additional installs needed, see Gymnasium documentation for details.

To test on different gym environments you can install the optional dependencies for them, e.g.

```
pip install .[dm-control]
pip install .[mani-skill2]
pip install .[brax]
pip install .[gymnax]
```