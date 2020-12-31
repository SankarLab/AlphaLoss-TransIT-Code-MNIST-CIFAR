#!/bin/bash
StringVal="1e-2"
C1="0"
C2="6"
Noise="0.0 0.1 0.2 0.3 0.4"
Momentum="0.0"
WD="0.0"
for lr in $StringVal; do
    for noise in $Noise; do
        mkdir mnist-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        cp mnist_master.sh mnist-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        cp mnist-noise-alpha.py mnist-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        cp mnist-template.sh mnist-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        cd mnist-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        bash mnist_master.sh $lr $1 $2 $noise $Momentum $WD
        cd ..
    done
done
