# Simple Sweep
1. Define the sweep: Similar to Hydra we can use a yaml file. Not sure if W&B supports arg parsing of sweep settings. Also, similar to Hydra-zen it allows defining setting as a dictionary.
2. Initialise the sweep using `wandb.sweep` method.
3. Run the sweep agent using `wandb.agent` method. This is exactly the same as Hydra-zen.