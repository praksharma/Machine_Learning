import wandb
import random

wandb.login()

parameters_dict = {
    'optimizer': {
        'values': ['adam']
        },
    'fc_layer_size': {
        'values': [128, 256]
        },
    'dropout': {
          'values': [0.3, 0.4]
        },
    }
sweep_config = {
    'method': 'grid'
    }
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="sweep_test")

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
