for alpha in 0.9 0.99 1.1 1.2 1.5 2.0 2.5 3.0 3.5 4.0 5.0 6.0 7.0 8.0; do
    sbatch cifar-template.sh $alpha $1 $2 $3 $4 $5 $6
    echo $alpha
done
