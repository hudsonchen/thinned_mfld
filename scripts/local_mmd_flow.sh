for seed in {0..4}
do
  for thinning in false kt random rbm
  do
    for particle_num in 64 256 1024
    do
      for zeta in 0.0
        do
          /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --dataset mmd_flow --particle_num $particle_num --step_size 1.0 --noise_scale 3e-4 --step_num 15000 --thinning $thinning --kernel gaussian --zeta $zeta
        done
    done
  done
done

