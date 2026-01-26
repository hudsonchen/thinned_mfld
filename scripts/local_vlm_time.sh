seed=0
for kernel in sobolev
do
for g in 0
do
    for particle_num in 64 256 1024 4096
    do
    for kt_function in compresspp_kt
        do
        /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 150 --thinning rbm --kernel $kernel --zeta 1.0
    done
    done
done
done
