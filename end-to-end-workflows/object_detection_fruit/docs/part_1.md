# Part 1: Generate synthetic data with Omnvierse

## Install Dependencies

In this section you can generate your synthetic data using the Omniverse GUI or as a headless version in your local terminal. Either option requires an Omniverse install.

- [Install Omniverse Launcher](https://docs.omniverse.nvidia.com/prod_install-guide/prod_install-guide/overview.html#omniverse-install-guide)


## Omniverse Launcher & Code

- [Install Omniverse Code](https://docs.omniverse.nvidia.com/prod_workflows/prod_workflows/extensions/environment_configuration.html#step-2-install-omniverse-code) from the `Exchange` tab within Omniverse Launcher

## Generate data in Omniverse GUI

Copy the contents of the generate_data.py script into the Script Editor tab in the bottom section of the Code window. Press the RUn1 button or ctrl + Enter on your keyboard to load the scene in the Viewport. From there you can preview a single scene in the Replciator tab at the top by clicking Preview   or run the full script by clicking Run. If you make no changes to this script it will generate 100 frames.

- From inside the Code GUI using the [script editor](https://docs.omniverse.nvidia.com/app_code/prod_extensions/ext_script-editor.html)
- If using Linux, copy code from `generate_data_gui.py` into the Script Editor window
-Execute code by clicking the `Run` button or pressing `ctrl+Enter`
- To preview what the scene will look like click Replicator then `Preview` in the top bar of your Omniverse Code window
- When you are ready to generate all your data go ahead and click `Replicator` and then `Run`, this will generate the designated number of frames and drop the RGB, bounding box data, and labels into the desired folder


## Generate data headlessly

Follow the documentation guidelines to launch a terminal in the correct folder location. The correct script to pass to your --/omni/replicator.scrip is generate_data_headless.py. This will generate and save the synthetic data in the same way as before, without utilizing the Omniverse GUI.

- [How to run](https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/headless_example.html) 
 - Script location: `/FruitBasketOVEReplicatorDemo/data_generation/code/generate_data_headless.py`
 - We need to locate `omni.code.replicator.sh` 
To find look for where Omniverse ode is locally installed
 - Run (script dictates where the output data is stored):
`./omni.code.replicator.sh  --no-window --/omni/replicator/script= “/FruitBasketOVEReplicatorDemo/data_generation/code/generate_data_headless.py”`
