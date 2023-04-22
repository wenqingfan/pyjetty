#! /bin/bash

#SBATCH --job-name="Pythia8HepMC"
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --partition=std
#SBATCH --time=24:00:00
#SBATCH --array=1-5000
#SBATCH --output=/rstorage/alice/AnalysisResults/wenqing/slurm-%A_%a.out

# Number of events per pT-hat bin (for statistics)
NEV_DESIRED=1000000

# Lower edges of the pT-hat bins
PTHAT_BINS=(7 9 12 16 21 28 36 45 57 70 85 99 115 132 150 169 190 212 235 260)
echo "Number of pT-hat bins: ${#PTHAT_BINS[@]}"

# Currently we have 8 nodes * 20 cores active
NCORES=5000
NEV_PER_JOB=$(( $NEV_DESIRED * ${#PTHAT_BINS[@]} / $NCORES ))
echo "Number of events per job: $NEV_PER_JOB"
NCORES_PER_BIN=$(( $NCORES / ${#PTHAT_BINS[@]} ))
echo "Number of cores per pT-hat bin: $NCORES_PER_BIN"

BIN=$(( ($SLURM_ARRAY_TASK_ID - 1) / $NCORES_PER_BIN + 1))
CORE_IN_BIN=$(( ($SLURM_ARRAY_TASK_ID - 1) % $NCORES_PER_BIN + 1))
PTHAT_MIN=${PTHAT_BINS[${PTHAT_BIN}]}
if [ $BIN -lt ${#PTHAT_BINS[@]} ]; then
	PTHAT_MAX=${PTHAT_BINS[$BIN]}
	echo "Calculating bin $BIN (pThat=[$PTHAT_MIN,$PTHAT_MAX]) with core number $CORE_IN_BIN"
else
	echo "Calculating bin $BIN (pThat_min=$PTHAT_MIN) with core number $CORE_IN_BIN"
fi

SEED=$(( ($CORE_IN_BIN - 1) * NEV_PER_JOB + 1111 ))

BASE_DIR=/software/users/wenqing/pyjetty/pyjetty/alice_analysis/generation
PYTHIA_DIR=$BASE_DIR/pythia8
OUTDIR="/rstorage/generators/pythia_alice/tree_gen/$SLURM_ARRAY_JOB_ID/$BIN/$CORE_IN_BIN"
mkdir -p $OUTDIR

# Load modules
module use /home/software/users/ploskon/heppy/modules
module load heppy/1.0
module use /home/software/users/wenqing/pyjetty/modules
module load pyjetty/1.0
module list

# Run PYTHIA
echo "Start time..."
date
echo "Running PYTHIA8 with the following settings..."
cat $PYTHIA_DIR/settings_${BIN}.cmnd
echo "Random seed:" ${SEED}
python $PYTHIA_DIR/pythia_gen_write_hepmc.py --py-cmnd $PYTHIA_DIR/settings_${BIN}.cmnd --nev $NEV_PER_JOB --py-seed $SEED -o $OUTDIR

# Convert hepmc to ntuple
python $BASE_DIR/hepmc2antuple_tn.py -i $OUTDIR/pythia8.hepmc -o $OUTDIR/AnalysisResultsGen.root --hepmc 2 -g pythia --nev $NEV_PER_JOB --no-progress-bar
echo "End time..."
date

echo "Remove hepmc output (to minimize output size)"
rm $OUTDIR/pythia8.hepmc

# Clean up
rm *.log

# Move stdout to appropriate folder
mv /rstorage/alice/AnalysisResults/wenqing/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /rstorage/alice/AnalysisResults/wenqing/${SLURM_ARRAY_JOB_ID}/
