#!/bin/bash

# Define the parameters to be changed
densities=('0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14')
betas=('0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0' '1.1' '1.2' '1.3' '1.4' '1.5' '1.6' '1.7' '1.8' '1.9' '2.0')
# Define the template file and the tmp file
template_file="slurm_template.slurm"
tmp_file_1="tmp_1_slurm_template.slurm"
tmp_file_2="tmp_2_slurm_template.slurm"

# Create and launch all the slurm jobs
echo Launching the Slurm jobs...

for density in "${densities[@]}";
do
	for beta in "${betas[@]}";
	do
		echo $beta
		# Gnerate the specific slurm instruction set and save it in a tmp file
		cp $template_file $tmp_file_1
		cp $template_file $tmp_file_2
		sed -e 's/DENSITY/'$density'/g' $template_file > $tmp_file_1
		sed -e 's/BETA/'$beta'/g' $tmp_file_1 > $tmp_file_2
		# Use the tmp file to launch the Slurm instance
		sbatch $tmp_file_2

		# Remove the tmp file
		rm $tmp_file_1
		rm $tmp_file_2

	done
done