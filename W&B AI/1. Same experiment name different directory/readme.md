There are several advantage of W&B over mlflow.
1. The logging depends on the name of experiment.
2. The storage is always a server. So, if you restart an unfinished experiment, it will continue to store things under same experiment name. Mlflow doesn't do this OOTB and sometimes it can be really tricky to handle a lot of stuffs.
3. If we run a training loop, we can simple log the individual metrics such as loss inside the training loop and it will create a plot for you. This is similar to Mlflow.
