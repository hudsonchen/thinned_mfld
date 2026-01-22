for seed in {0..9}
do
  for kernel in sobolev
  do
    for g in 0 1 2
    do
      for particle_num in 16 64 256 1024 4096
      do
        for kt_function in compress_kt
          do
            python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 200 --thinning kt --kernel $kernel --zeta 0.1 --kt_function $kt_function --save_path './results/'
          done
      done
    done
  done
done