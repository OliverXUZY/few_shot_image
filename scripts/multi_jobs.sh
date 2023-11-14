
# for M in 200 1000
# do
#     for num_shot in 1 10 20
#     do
#         sbatch script.sh $M $num_shot "200"
#     done
# done


# shot_list=(1 2 3 4)
# m_list=(150 300 450 600)

# for M in 600 800
# do
#     for i in "${!shot_list[@]}"
#     do
#         shot=${shot_list[i]}
#         m=${m_list[i]}
#         sbatch --exclude=euler24,euler25,euler26,euler27 subVision.sh $M $shot $m
#     done
# done

for M in 200 400 600 800
do
    sbatch --array=1-4 jobArrayScript.sh $M
done