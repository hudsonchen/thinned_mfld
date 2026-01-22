# Create and activate conda environment
conda create -y -n thin_mfld_3 python=3.10
conda activate thin_mfld_3

# Install dependencies
pip install --upgrade pip
pip install jax numpy matplotlib goodpoints jax_tqdm optax jaxtyping scikit-learn diffrax
pip install qpsolvers[open_source_solvers]


# Run experiment
python run_mfld.py --seed 0 --dataset mmd_flow --particle_num 1024 --step_size 1.0 --noise_scale 3e-4 --step_num 15 --thinning kt --kernel gaussian --zeta 0.1 --g 0

echo "Environment installed and tested!"