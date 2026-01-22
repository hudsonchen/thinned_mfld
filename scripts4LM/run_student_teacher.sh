for seed in {0..10}
do
  for d in 10
    do
      for g in 0 1 2
      do
      for particle_num in 64 256 1024 4096
      do
      for kt_function in compress_kt
      do
      python run_mfld.py --dataset student_teacher --seed $seed --g $g --particle_num $particle_num --step_size 0.01 --noise_scale 0.0 --bandwidth 1.0 --step_num 100 --thinning kt --kernel gaussian --zeta 0.0001 --d $d --teacher_num 100 --kt_function $kt_function
      done
      done
    done
    done
done