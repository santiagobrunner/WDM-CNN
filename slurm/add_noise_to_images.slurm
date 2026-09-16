#!/bin/bash -l
#SBATCH --job-name="input_image_test" 
#SBATCH --time=02:00:0 --mem=32G 
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --output=%j.o 
#SBATCH --error=%j.e 
#======START===== 
module list
module load apptainer
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo $OMP_NUM_THREADS
shopt -s nullglob

HALO=$1
SB_LIMIT=$2
FILTER=$3
CONTAINER="/shares/feldmann.ics.mnf.uzh/Santiago/Thesis/conda.sif"
INPUT_BASE="/shares/feldmann.ics.mnf.uzh/Santiago/input_images_${FILTER}"
OUTPUT_BASE="/shares/feldmann.ics.mnf.uzh/Santiago/input_images_${FILTER}_${SB_LIMIT}_f"

echo 'Start:' "$(date)  halo=$HALO  SB_limit=$SB_LIMIT"



mkdir -p "${OUTPUT_BASE}/${HALO}"

for SNAP_DIR in "${INPUT_BASE}/${HALO}"/*/; do
	SNAP=$(basename "$SNAP_DIR")
	mkdir -p "${OUTPUT_BASE}/${HALO}/${SNAP}"
		
	for FILENAME in ${INPUT_BASE}/${HALO}/${SNAP}/*.fits.gz; do
		OUT_BASENAME=$(basename $FILENAME)
		OUTPUT_FILE="${OUTPUT_BASE}/${HALO}/${SNAP}/${OUT_BASENAME%.fits.gz}_N${SB_LIMIT}.fits"
		[ -f "${OUTPUT_FILE}.gz" ] && continue
		echo apptainer exec $CONTAINER mockimgs_sb_add_noise $FILENAME --SB_limit $SB_LIMIT --SB_SN 3 --output-flux -o $OUTPUT_FILE
		apptainer exec $CONTAINER mockimgs_sb_add_noise $FILENAME --SB_limit $SB_LIMIT --SB_SN 3 --output-flux -o $OUTPUT_FILE
	done
done
echo 'End:' "$(date)"	

