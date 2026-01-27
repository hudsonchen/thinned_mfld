seed=0
g=0
for kernel in sobolev
do
for method in random rbm false
do
    for particle_num in 64 256 1024 4096
    do
    for kt_function in compresspp_kt
        do
        /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 150 --thinning rbm --kernel $kernel --zeta 1.0 --save_path './results_time'
    done
    done
done
done

seed=0
for kernel in sobolev
do
for g in 0 1 2
do
    for particle_num in 64 256 1024 4096
    do
    for kt_function in compress_kt
        do
        /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 150 --thinning kt --kernel $kernel --zeta 1.0 --kt_function $kt_function --skip_swap --save_path './results_time'
        /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 150 --thinning kt --kernel $kernel --zeta 1.0 --kt_function $kt_function --save_path './results_time'
    done
    done
done
done