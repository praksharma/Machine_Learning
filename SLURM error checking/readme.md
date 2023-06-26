# SLURM error checks

This directory contains files to check for errors when running a large number of SLURM jobs. I often breakdown the number of hyperparameters to separate jobs as there is a time limit on GPU nodes (generally 2 days).

Assume, I have 75 different jobs created from hyperparameter breakdown such that none of them exceeds the time limit. Unfortunately, some of them will exceed the timelimit. There two easy ways to check for errors in all 75 folders.

* Write custom script for email if the jobs script fails.
* Write a Python script to search for time limit error in each folder.

## Write custom script for email if the jobs script fails.

```sh
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name NVIDIA_2D_heat
#SBATCH -o batch_output.log
#SBATCH -e batch_error.log
#SBATCH --gres=gpu:1
#SBATCH --account=scw1901
#SBATCH --partition=accel_ai
#SBATCH --mail-type=FAIL

echo "Job script path: ${SLURM_SUBMIT_DIR}"| mail -s "Job Failure: ${SLURM_JOB_NAME}" -r 1915438@swansea.ac.uk 1915438@swansea.ac.uk

# Your jobs here
sleep 5 # allow mail to do preprocessing for short jobs
exit 1 # intentionally generate a SLURM error

echo "Script completed successfully"
```

Here we need to remove the ```#SBATCH --mail-user=1915438@swansea.ac.uk``` line otherwise we will get the default email with default subject which looks like this ```Slurm Job_id=7549700 Name=NVIDIA_2D_heat Failed, Run time 00:00:01, FAILED, ExitCode 1```. So we will end up with 2 emails.

# Write a Python script to search for time limit error in each folder
 I've prepared a python script in case you want to find out failed jobs without accessing to the emails. Just run the ```finder.py``` and modify filenames in ```main()```.