for seed in {2..5}
do
  for d in 10
    do
      for g in 0 1 2
      do
<<<<<<< HEAD
      for particle_num in 64 256 1024 4096
=======
      for particle_num in 64 256 1024
>>>>>>> 87c6e2d5e0940a62ba829a3a411aa819bdfab16f
      do
      for kt_function in compress_kt
      do
      /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --dataset student_teacher --seed $seed --g $g --particle_num $particle_num --step_size 0.01 --noise_scale 0.0 --bandwidth 1.0 --step_num 100 --thinning kt --kernel gaussian --zeta 0.0001 --d $d --teacher_num 100 --kt_function $kt_function --skip_swap
      done
      done
    done
    done
done

for seed in {0..5}
do
  for thinning in kt
  do
<<<<<<< HEAD
    for particle_num in 64 256 1024 4096
=======
    for particle_num in 64 256 1024
>>>>>>> 87c6e2d5e0940a62ba829a3a411aa819bdfab16f
    do
      for zeta in 0.0
        do
        for g in 1 2
        do
        for kt_function in compress_kt
        do
          /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset mmd_flow --particle_num $particle_num --step_size 1.0 --noise_scale 3e-4 --step_num 15000 --thinning $thinning --kernel gaussian --zeta $zeta --g $g --kt_function $kt_function --skip_swap
        done
        done
        done
    done
  done
done

<<<<<<< HEAD
for seed in {0..5}
do
  for kernel in sobolev
  do
    for g in 0 1 2
    do
      for particle_num in 4096
      do
        for kt_function in compress_kt
          do
            /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 200 --thinning kt --kernel $kernel --zeta 0.1 --kt_function $kt_function --skip_swap
          done
      done
    done
  done
done
=======
# for seed in {0..5}
# do
#   for kernel in sobolev
#   do
#     for g in 0 1 2
#     do
#       for particle_num in 4096
#       do
#         for kt_function in compress_kt
#           do
#             /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 200 --thinning kt --kernel $kernel --zeta 0.1 --kt_function $kt_function --skip_swap
#           done
#       done
#     done
#   done
# done
>>>>>>> 87c6e2d5e0940a62ba829a3a411aa819bdfab16f
