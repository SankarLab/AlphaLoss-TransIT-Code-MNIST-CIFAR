#!/bin/bash
StringVal="5e-2"
C1="4"
C2="6"
Noise="0.1 0.2 0.3 0.4"
Momentum="0.0"
WD="0.0"
for lr in $StringVal; do
    for noise in $Noise; do
        mkdir cifar-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        cp cifar_master.sh cifar-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        cp cifar-noise-alpha.py cifar-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        cp cifar-template.sh cifar-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        cd cifar-imb-$1-$2-binary-$C1-$C2-noise-$noise-lr_$lr
        bash cifar_master.sh $lr $1 $2 $noise $Momentum $WD
        cd ..
    done 
done
