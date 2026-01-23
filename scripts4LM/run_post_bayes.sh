for seed in {2..9}
do
  for method in random rbm false
  do
    for g in 0
    do
      for particle_num in 16 64 256 1024
      do
        taskset -c $((($seed - 2)*12))-$(((($seed - 2)*12)+11)) python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 150 --thinning $method --kernel sobolev --zeta 0.1
      done
    done
  done &
done


for seed in {0..9}
do
  for method in false
  do
    for g in 0
    do
      for particle_num in 16 64 256 1024
      do
        python run_mfld.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 150 --thinning $method --kernel sobolev --zeta 0.1
      done
    done
  done
done