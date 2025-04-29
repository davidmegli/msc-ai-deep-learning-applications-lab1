## David Megli - Deep Learning Applications Lab 1

## Experiment 1.2
1. Edit network and training configurations in "base_config.yaml". Set use_wandb to False if needed.
2. Run "run_experiments.py". By default it will train a MLP and a Residual MLP, at different depths (2, 4, 8, 16, 32)
3. Wait
4. The script will save a "result.csv" file and 2 plot (.png) visualizing the comparison of loss and accuracy at different depths, for both networks. The files will be located in the "outputs/experiments" folder.

## Experiment 1.3
1. Edit network and training configurations in "base_cnn_config.yaml". Set use_wandb to False if needed.
2. Run "run_cnn_experiments.py". By default it will train a CNN and a Residual MLP uses ResNet BasicBlock, at different depths (2, 4, 8, 16, 32)
3. Wait
4. The script will save a "result.csv" file and 2 plot (.png) visualizing the comparison of loss and accuracy at different depths, for both networks. The files will be located in the "outputs/cnn_experiments" folder.