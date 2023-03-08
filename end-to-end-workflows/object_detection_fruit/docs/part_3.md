# Part 3: Deploy model to Triton 

## Start triton server
When we start the server we want our model to be properly located in the `/models/fasterrcnn_resnet50/1` folder.

`sudo docker run --gpus=1 --rm -p9000:8000 -p9001:8001 -p9002:8002 -v /home/$USER/sdg_workflow/models/:/models nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/models`

Once started, you should see:
```
+---------------------+---------+--------+
| Model               | Version | Status |
+---------------------+---------+--------+
| fasterrcnn_resnet50 | 1       | READY  |
+---------------------+---------+--------+
```

## Start triton client

In another terminal window, with your server running start your client

- `sudo docker run -it --rm --net=host -v /home/zoe/Desktop/sdg_workflow:/workspace nvcr.io/nvidia/tritonserver:23.01-py3-sdk`

- To use the deploy script you can see required parameters by running
 - `python deploy.py --help`

- Example command:
 - ` python deploy.py -p /workspace/rgb_0.png`

