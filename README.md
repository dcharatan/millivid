<h1 align="center">MilliVid: Hierarchical Latents for Long-Range Consistency in Video Generation</h1>

<p align="center">
  <a href="https://ishaanchandratreya.github.io/">Ishaan Preetam Chandratreya</a><sup>*,1</sup>,
  <a href="https://davidcharatan.com/">David Charatan</a><sup>*,1</sup>,
  <a href="https://basile.be/about-me/">Basile Van Hoorick</a><sup>2</sup>,
  <a href="https://zakharos.github.io/">Sergey Zakharov</a><sup>2</sup>,
  <a href="https://vitorguizilini.github.io/">Vitor Guizilini</a><sup>2</sup>,
  <a href="https://web.mit.edu/phillipi/">Phillip Isola</a><sup>1</sup>,
  <a href="https://www.vincentsitzmann.com/">Vincent Sitzmann</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Massachusetts Institute of Technology&nbsp;&nbsp;
  <sup>2</sup>Toyota Research Institute
</p>

<p align="center">
  <sup>*</sup>equal contribution
</p>

<h3 align="center">arXiv 2026</h3>

<p align="center">
  <a href="https://davidcharatan.com/millivid/">[Project Page]</a>
  <a href="https://arxiv.org/abs/2606.09056">[Paper]</a>
  <a href="https://github.com/dcharatan/millivid">[Code]</a>
  <a href="https://huggingface.co/charatan/millivid">[Models]</a>
  <a href="https://huggingface.co/datasets/charatan/loopcraft">[Dataset]</a>
</p>

> **TL;DR:** We present a long-memory autoregressive framework for video generation. A hierarchical latent space and coarse-to-fine rollout let it stay consistent many times longer than a conventional diffusion model under the same token budget.

## Abstract

Video generative models have become increasingly powerful, but long-range consistency remains challenging to achieve because even a few dozen frames require impractically long transformer sequence lengths. We show that this issue can be mitigated by generating video using coarse-to-fine rollout within a multi-scale token space. Our approach is simple: first, we pre-train an autoencoder that compresses each frame into a hierarchy of tokens, with levels ranging from the typical latent resolution to only a handful of tokens per frame. The coarsest levels capture the most consequential information—such as scene layout and semantics—while finer levels add high-frequency appearance and texture. Then, we train a video diffusion model to generate these tokens using coarse-to-fine rollout. By carefully controlling the level of detail at which frames are generated and used as context during each rollout step, we are able to preserve long-range consistency in geometry and object permanence while spending less compute on the long-range consistency of less perceptually relevant details. We validate this approach using a custom dataset of long Minecraft videos, where it produces substantially more consistent rollouts compared to existing baselines.

## Quick Start

### Setup

#### 1. Create a Conda environment and install dependencies.

```bash
conda create python=3.12 -n millivid
conda activate millivid
pip install -r requirements.txt
```

#### 2. Generate some videos!

Use the command below to generate videos with a pretrained MilliVid model. It will automatically download a tiny subset of the Loopcraft dataset and a pretrained MilliVid model.

```bash
python3 demo.py
```

If you want to compare MilliVid's outputs to the baselines' outputs, use `python3 demo.py --baselines` instead.

## Code Release Progress

- ✅ Pre-trained checkpoints (for main baseline comparisons)
- ✅ Demo/inference script
- ✅ Images and latents for test set uploaded
- ⏳ Images and latents for training set are currently uploading
- ❌ Training/testing instructions not yet given (will be done over the next few days)
- ❌ Data generation scripts not yet uploaded (coming soon as well)

TL;DR: If you're very eager to start experimenting with MilliVid, everything you need is here, but there may be some sharp edges. Those will be ironed out in the coming days.

## Useful Information for Extending MilliVid

- [Experiment Configuration README:](config/experiment/README.md) Configurations for various models and ablations.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{chandratreya2026millivid,
  title     = {MilliVid: Hierarchical Latents for Long-Range Consistency in Video Generation},
  author    = {Chandratreya, Ishaan Preetam and Charatan, David and Van Hoorick, Basile and Zakharov, Sergey and Guizilini, Vitor and Isola, Phillip and Sitzmann, Vincent},
  booktitle = {arXiv},
  year      = {2026},
  url       = {https://davidcharatan.com/millivid},
}
```

## Acknowledgements

We thank Andrew Song and Hannah Schlueter for their feedback during the process of writing and editing the paper. This work was supported by the Toyota Research Institute (TRI) University 3.0 (URP) program, the National Science Foundation under Grant No. 2211259, by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) under 140D0423C0075, by the Amazon Science Hub, by the MIT-Google Program for Computing Innovation, by AMD via the MIT AI Hardware Program, and by a 2025 MIT Office of Research Computing and Data Seed Grant. The views and conclusions contained in this document are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of any other entity.
