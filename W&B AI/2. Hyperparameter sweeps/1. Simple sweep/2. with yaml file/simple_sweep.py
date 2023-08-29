import wandb
import random
import yaml
from pathlib import Path

wandb.login()

sweep_config = yaml.safe_load(Path("conf.yaml").read_text())
sweep_id = wandb.sweep(sweep_config, project="sweep_test_1")

def train(config=None):

    with wandb.init(config=config):
        config = wandb.config
        
        print("Optimiser: ",config.optimizer)
        print("Number of layers: ",config.fc_layer_size)
        print("Dropout: ", config.dropout)
        
        for epoch in range(1000):  # loop over the dataset multiple times
            loss = random.random()
            wandb.log({'epoch': epoch + 1, 'loss': loss})

        print('Finished Training')
        
wandb.agent(sweep_id, function=train)#, count=5)
wandb.finish()
