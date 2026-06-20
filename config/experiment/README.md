# Experiment Configurations

The experiment configurations are split into three categories:

- `main_*`: These are the main baseline comparisons.
- `ablation_*`: These are for ablations.
- `eval_*`: These are used for evaluation. See the notes below.

To use an experiment, append `+experiment=blah` or `+experiment=[blah1,blah2]` to the script you're trying to run. For example:

```bash
python3 -m train +experiment=main_millivid
```

## Eval Configurations

Here's what the eval configurations do:

- `eval_autoencoder_test_set.yaml`: Used to point the autoencoder at the test set (e.g., for encoding test set latents).
- `eval_fake_model_ground_truth_latents.yaml`: Used to evaluate a fake model that outputs ground-truth video latents. This is used as an "upper bound" in our plots.
- `eval_fake_model_random_latents.yaml`: Used to evaluate a fake model that outputs randomly ordered ground-truth video latents. This is used as a "lower bound" in our plots. Note that models that suffer from exposure bias may perform worse than randomly ordered latents on many metrics.
- `eval_test_set_cascaded.yaml`: Used in conjunction with other experiments to point models at the cascaded test set latents.
- `eval_test_set.yaml`: Used in conjunction with other experiments to point models at the test set latents (i.e., the regular hierarchical ones).
