for seed in {4..5}
do
  for d in 10
    do
      for thinning in false kt random rbm
      do
      for particle_num in 64 256 1024 4096
      do
      for teacher_num in 100 1000
      do
      /home/zongchen/miniconda3/envs/thin_mfld/bin/python run_mfld.py --dataset student_teacher --seed $seed --g 0 --particle_num $particle_num --step_size 0.01 --noise_scale 0.0 --bandwidth 1.0 --step_num 100 --thinning $thinning --kernel gaussian --zeta 0.0001 --d $d --teacher_num $teacher_num
      done
      done
    done
    done
done

# for seed in 0 1 2
# do
#   for thinning in random false
#   do
#     for particle_num in 16 64 1024
#     do
#     /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --dataset covertype --seed $seed --particle_num $particle_num --step_size 0.01 --noise_scale 0.001 --step_num 50 --thinning $thinning --kernel sobolev
#     done
#   done
# done