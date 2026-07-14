<!-- source: https://docs.rc.asu.edu/slurm-sbatch -->
# Slurm SBATCH Job Scripts | ASU RC Docs

SBATCH scripts are bash scripts with `#SBATCH` header directives interpreted by Slurm before execution.

## Job Manipulation

### Submitting Jobs
```bash
sbatch <scriptName>
```
Override header options at submission: `sbatch -c 4 -t 1-0 my_script.sh`

### Updating a Pending Job
```bash
scontrol update job <jobID> <jobfield> <new value>
scontrol update job 11254871 ReqCores=4
scontrol update job 11254871 QOS=private Partition=htc
scontrol update job 11254871 Gres=gpu:a100:2
```

### Canceling Jobs
```bash
scancel <jobID>
```

### MPI (Parallel) Job
```bash
#!/bin/bash
#SBATCH -N 3
#SBATCH -n 8
#SBATCH -c 1
#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE

module load openmpi/4.1.5
module load lammps

srun -n 8 --mpi=pmix software
```

### Job Arrays
```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=1-20
#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE

readonly manifest="/path/to/manifest_file"
readonly run_opts=($(getline $SLURM_ARRAY_TASK_ID "$manifest"))

module load mamba/latest
source activate myEnv
python myscript.py "${run_opts[@]}"
```
See [Slurm Job Array Examples](/slurm-job-array-examples).

## Troubleshooting

- **DOS line breaks**: `sbatch: error: Batch script contains DOS line breaks (\r\n)` — fix with `dos2unix myjob.sh`.
- **Missing shebang**: `sbatch: error: This does not look like a batch script...` — first line must be `#!/bin/bash`.
- **Invalid feature specification**: check partition/QoS/constraints against [Partitions and QoS](/partitions-and-qos).

### Slurm Exit Codes
- 0 → success
- non-zero → failure
- 1 → general failure
- 2 → incorrect use of shell builtins
- 3-124 → some error in job
- 125 → out of memory
- 126 → command cannot execute
- 127 → command not found
- 128 → invalid argument to exit
- 129-192 → terminated by Linux signal (subtract 128 to get signal number; `kill -l` lists codes)
