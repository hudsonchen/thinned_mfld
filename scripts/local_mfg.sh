for seed in {0..4}
do
<<<<<<< HEAD
  for thinning in random
  do
    for particle_num in 4096
=======
  for thinning in false kt random
  do
    for particle_num in 256 1024 4096
>>>>>>> 87c6e2d5e0940a62ba829a3a411aa819bdfab16f
    do
        /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfg.py --thinning $thinning --relax 0.05 --bandwidth 1.0 --step_num 100 --particle_num $particle_num --kernel gaussian --dt 0.01 --seed $seed
    done
  done
done

<<<<<<< HEAD

for seed in {0..4}
do
  for thinning in kt
  do
    for particle_num in 4096
    do
    for kt_function in compress_kt
    do
        /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfg.py --thinning $thinning --relax 0.05 --bandwidth 1.0 --step_num 100 --particle_num $particle_num --kernel gaussian --dt 0.01 --seed $seed --kt_function $kt_function --skip_swap
    done
    done
  done
done


for seed in {0..4}
do
  for thinning in kt
  do
    for particle_num in 4096
    do
    for kt_function in compresspp_kt
    do
        /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfg.py --thinning $thinning --relax 0.05 --bandwidth 1.0 --step_num 100 --particle_num $particle_num --kernel gaussian --dt 0.01 --seed $seed --kt_function $kt_function
    done
    done
  done
done
=======
>>>>>>> 87c6e2d5e0940a62ba829a3a411aa819bdfab16f
