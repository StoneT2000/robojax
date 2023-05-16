# Contributing to Robojax

Thanks for contributing to Robojax! To get started make sure to setup your development environment properly.

```
git clone https://github.com/StoneT2000/robojax
conda env create -f environment.yml
conda activate robojax
```

To maintain code quality and standards, this repo uses [pre-commit](https://pre-commit.com/).

Install it and set it up by running
```
pre-commit
```

To test on different gym environments, to ensure no conflicts different conda envs are recommended. First run

```
conda env create -n "robojax_dmc" -f environment.yml
```

Then run #TODO