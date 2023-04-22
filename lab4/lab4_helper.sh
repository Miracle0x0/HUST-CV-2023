#!/bin/zsh

conda activate torch

echo "poison_ratio: 0.0"
python ./tools/train.py ./configs/vgg13_bn_00.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.1"
python ./tools/train.py ./configs/vgg13_bn_01.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.2"
python ./tools/train.py ./configs/vgg13_bn_02.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.3"
python ./tools/train.py ./configs/vgg13_bn_03.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.4"
python ./tools/train.py ./configs/vgg13_bn_04.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.5"
python ./tools/train.py ./configs/vgg13_bn_04.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.6"
python ./tools/train.py ./configs/vgg13_bn_05.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.7"
python ./tools/train.py ./configs/vgg13_bn_06.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.8"
python ./tools/train.py ./configs/vgg13_bn_07.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 0.9"
python ./tools/train.py ./configs/vgg13_bn_08.py --work_dir './work_dir' --save_dir './models/' --seed 1
echo "poison_ratio: 0.0"

python ./tools/train.py ./configs/vgg13_bn_09.py --work_dir './work_dir' --save_dir './models/' --seed 1

echo "poison_ratio: 1.0"
python ./tools/train.py ./configs/vgg13_bn_10.py --work_dir './work_dir' --save_dir './models/' --seed 1
