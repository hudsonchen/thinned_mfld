for seed in {0..4}
do
  for thinning in false kt random
  do
    for particle_num in 256 1024
    do
        /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfg.py --thinning $thinning --relax 0.05 --bandwidth 1.0 --step_num 100 --particle_num $particle_num --kernel gaussian --dt 0.01 --seed $seed
    done
  done
done

