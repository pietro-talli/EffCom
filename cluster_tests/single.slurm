#!/bin/sh
#SBATCH --job-name=EffCom_1_0.9_push
#SBATCH --output=../logs/ASoI_1_0.9_%j.log
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=30-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tallipietr@dei.unipd.it

env=eff_com
pydir=/nfsd/signet4/tallipietr/miniconda3/envs/$env/bin

cd ..

echo
echo Starting the job...

$pydir/python3 test_estimation_30.py \
--density 1 \
--beta 0.9

