for seed in {0..4}
do
  for kernel in gaussian
    do
      for g in 0
      do
      for particle_num in 64 256 1024
      do
      for noise_scale in 0.01
      do
      /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --dataset covertype --seed $seed --g $g --particle_num $particle_num --step_size 0.01 --noise_scale $noise_scale --bandwidth 1.0 --step_num 50 --thinning kt --kernel $kernel --zeta 0.0001
      done
      done
    done
    done
done

for seed in {0..4}
do
  for thinning in random false rbm
  do
    for particle_num in 64 256 1024
    do
    for noise_scale in 0.01
    do
    /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --dataset covertype --seed $seed --particle_num $particle_num --step_size 0.01 --noise_scale $noise_scale --step_num 50 --thinning $thinning --kernel sobolev --zeta 0.0001
    done
    done
  done
done