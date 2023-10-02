# NYU HPC Greene Demo

The very first thing is that you need to be connected to NYU VPN. You need download Anyconnect from Cisco. Further information on the installation and setup of NYU VPN can be found [here](https://nyu.service-now.com/sp?id=kb_article&sysparm_article=KB0011177&sys_kb_id=6177d7031c811904bbcf4dc2835ec340&spa=1). 

Once you are on the network, you can log in to the gateway servers using the shell command:

```bash
ssh <NYU-NET-ID>@gw.hpc.nyu.edu
```

The next step is to get you connected to NYU greene HPC:

```bash
ssh <NYU-NET-ID>@greene.hpc.nyu.edu
```

to access the burst node, you can run:

```bash
ssh burst
```

from the greene node.

**Note:**

You are not allowed to download the packages in HPC, so as a result they have already provided set of packages and you can use them 

To check all the available packages on NYU HPC:

```bash
module avail
```

**for example:**

- For the assignment we were asked to use MKL library, the package that we had use was: `python/intel/3.8.6/` 

- To use that package, you will write:

```bash
module load python/intel/3.8.6
```

- You can also check the list of packages that you are currently using:

```bash
module list
```

- For resetting all the packages that you have loaded, you can use the following command:

```bash
module purge
# wait for the output
module list
# to check if the packages have been purged out or not
```

and this will give you a fresh restart where you can load new packages for your programs.

## Executing the code on NYU HPC:

Sample Slurm Job Script:

```bash
#!/bin/bash
#SBATCH --job-name=my_c_program
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --output=my_program_output.txt

./your_program
```

Save the script as something like `job_script.sh`.

**Submit Your Job**:

```bash
sbatch job_script.sh
```

**Monitor Job Status**: You can check the status of your job with:

```bash
squeue -u your_netid
```

<mark>From the video:</mark>

Slurm Job Script:

```bash
#!/bin/bash
#SBATCH --job-name=hpml
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out
#SBATCH --mem=4GB
#SBATCH --time=00:10:00


module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate idls
cd /scratch/pm3483/name_of_directory


python code.py
```

`#SBATCH --job-name=hpml`

It means the task or job name that you give. You can give any name as per your wish

`#SBATCH --cpus-per-task=1`

Assign number of CPUs as per your requirement 

`#SBATCH --output=%x.out`

%x means that it is same as job name, meaning that the output file would be hpml.out in this case. It is a output file, whatever print you will get from running of the python file.

`#SBATCH --mem=4GB`

This means, I am assigning 4 gb of RAM, which is more than enough to run the program

`#SBATCH --time=00:10:00`

This means, how long you want to run this program. This will run the program for 10 minutes.
