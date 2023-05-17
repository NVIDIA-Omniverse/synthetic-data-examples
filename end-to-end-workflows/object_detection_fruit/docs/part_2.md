# Part 2: Training a model with synthetic data

## Setup training
To use the training script you can see required parameters by running
 -`python train.py --help`

- Example command:
 - `python train.py -d /home/omni.replicator_out/fruit_data_$DATE/ -o /home/model.pth -e 10`

## Visualize training
We have included a visdualization script to run after your first training. This will show how Omniverse generates the labeled data. To see required parameters
- `python visualize.py --help`

- Example command:
 - `python visualize.py -d /home/$USER/omni.replicator_out/fruit_data_$DATE -o /home/$USER -n 0`

## Export model
- To use the export script you can see required parameters by running
 - `python export.py --help`
- Example command, make sure to dave to the `models/fasterrcnn_resnet50/1`
 - `python export.py -d /home/out.pth -o /home/models/fasterrcnn_resnet50/1`
