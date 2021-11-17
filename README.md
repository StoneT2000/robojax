# PaperRL
High quality, super reproducible, implementations of reinforcement learning papers with as many direct references to text as possible

Features:
- Readable: No more variables like [surr](https://github.com/thu-ml/tianshou/blob/8f19a8696672fa49ead41e4e93d5d79855e5d6aa/tianshou/policy/modelfree/ppo.py#L119)
- Reproducible: Reproducibility plagues every field, especially RL where results are often very unstable. So **everything** is not only seeded, it's also made obvious how things are seeded, initialized, randomly generated etc.
- Paper references: This repository intends to reference papers / articles / equations etc. directly as much as possible. This can help you learn from code and text, not just text, and / or help you write your own RL code using all the tricks papers use without telling you about them.

## Setup

I highly recommend using conda. Otherwise you can try and install all the packages yourself (at your own risk of not getting reproducible results)

```
conda env create -f environment.yml
```
