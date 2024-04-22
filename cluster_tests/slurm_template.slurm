#!/bin/sh
#SBATCH --job-name=EffCom_DENSITY_BETA_push
#SBATCH --output=../logs/ASoI_DENSITY_BETA_%j.log
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

$pydir/python3 test_pull_estimation_30.py \
--density DENSITY \
--beta BETA

