
for M in 200 1000
do
    for num_shot in 1 10 20
    do
        sbatch script.sh $M $num_shot "200"
    done
done