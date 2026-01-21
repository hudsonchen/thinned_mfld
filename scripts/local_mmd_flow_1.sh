for seed in {0..4}
do
  for thinning in kt
  do
    for particle_num in 64 256 1024 4096
    do
      for zeta in 0.0
        do
        for g in 0
        do
        for kt_function in compresspp_kt
        do
          /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset mmd_flow --particle_num $particle_num --step_size 1.0 --noise_scale 3e-4 --step_num 15000 --thinning $thinning --kernel gaussian --zeta $zeta --g $g --kt_function $kt_function
        done
        done
        done
    done
  done
done

# for seed in {0..9}
# do
#   for thinning in false random
#   do
#     for particle_num in 64 256 1024 4096
#     do
#       for zeta in 0.0
#         do
#         for g in 0
#         do
#         for kt_function in compress_kt
#         do
#           /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset mmd_flow --particle_num $particle_num --step_size 1.0 --noise_scale 3e-4 --step_num 15000 --thinning $thinning --kernel gaussian --zeta $zeta --g $g --kt_function $kt_function
#         done
#         done
#         done
#     done
#   done
# done