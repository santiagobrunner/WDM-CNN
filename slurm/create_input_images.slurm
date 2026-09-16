#!/bin/bash -l
#SBATCH --job-name="input_image_test" 
#SBATCH --time=2:30:0 --mem=32G 
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --output=%j.o 
#SBATCH --error=%j.e 
#======START===== 
module list
module load apptainer
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo $OMP_NUM_THREADS
HALO_FILTER=$1
FILTER=$2

CONTAINER="/shares/feldmann.ics.mnf.uzh/Santiago/Thesis/conda.sif"
DATAPATH_BASE="/shares/feldmann.ics.mnf.uzh/Santiago/FIREbox_zoom_extracted"
DATAPATH_APPENDIX="to_pNbody"

INSTRUMENT="/shares/feldmann.ics.mnf.uzh/Santiago/Thesis/instruments/training_instrument_Euclid_${FILTER}.py"
LOS="/shares/feldmann.ics.mnf.uzh/Santiago/Thesis/hp_nside2_centers.txt"

OUTPATH_BASE="/shares/feldmann.ics.mnf.uzh/Santiago/input_images_${FILTER}"
mkdir -p "${OUTPATH_BASE}"

MOCKIMGSPATH="/shares/feldmann.ics.mnf.uzh/Santiago/Thesis/mockimgs_sb_compute_images"

# HALOS="FB15_h15_L11_r4_p1"
echo 'Start:' "$(date)"
for HALO_DIR in $DATAPATH_BASE/FB15_${HALO_FILTER}_L11_*; do
	HALO=$(basename $HALO_DIR)
	DATAPATH="${DATAPATH_BASE}/${HALO}/${DATAPATH_APPENDIX}"

	# Skip already processed halo
	#[[ "$HALO" == "FB15_h15_L11_r4_p1" ]] && continue
	

	for GAL_FILE in $DATAPATH/*.hdf5;do
		
		#Create correct Output name
		BASENAME=$(basename $GAL_FILE)		# pnbody_wdm07d5_100_00000_Rvir.hdf5
		NOEXT=${BASENAME%.hdf5}                 # pnbody_wdm07d5_100_00000_Rvir
		DM=$(echo $NOEXT | cut -d'_' -f2)                # wdm07d5
		SNAP=$(echo $NOEXT | cut -d'_' -f3)              # 100
		HALO_NR=$(echo $HALO | cut -d'_' -f2)
		OUTPUT="${OUTPATH_BASE}/${HALO}/${SNAP}/SB_${HALO_NR}_${DM}_${SNAP}.fits"
		
		#Skip if already exists
		[ -f "${OUTPUT%.fits}_LOS00.fits" ] && continue

		# Create output directory for this Halo if it doesn't exist
        	mkdir -p "${OUTPATH_BASE}/${HALO}/${SNAP}"	
			
		echo srun apptainer exec $CONTAINER python $MOCKIMGSPATH "$GAL_FILE" --instrument "$INSTRUMENT" --distance 35 --los_file "$LOS" --rsp_mode None --rsp_fac 0.6 -o "$OUTPUT"
		apptainer exec $CONTAINER python $MOCKIMGSPATH "$GAL_FILE" --instrument "$INSTRUMENT" --distance 35 --los_file "$LOS" --rsp_mode=None --rsp_fac 0.6 -o "$OUTPUT"
		
	done
done

echo 'End:' "$(date)"	

#=====END====

