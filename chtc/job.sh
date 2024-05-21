#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

pid=${1}
step=${2}
cmd=`tr '*' ' ' <<< $3`
echo $cmd

CODENAME=PROPS
export PATH

cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz -C .
rm ${CODENAME}.tar.gz # remove code tarball
cd $CODENAME

# install code
#ENVNAME=props
#ENVDIR=${ENVNAME}
#cp /staging/ncorrado/${ENVNAME}.tar.gz .
#mkdir $ENVDIR
#tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
#rm ${ENVNAME}.tar.gz # remove env tarball
#source $ENVDIR/bin/activate

python3 -m venv env
source env/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3
pip install tensorboard
pip install "mujoco<3"
pip install imageio
pip install gymnasium
pip install pyyaml

pip install -e .
pip install -e PROPS/custom-envs
cd PROPS

python3 -u ${cmd} --run-id ${step} --seed ${step}
#$($cmd --seed $step --run-id $step)
#$($command)

tar -czvf results_${pid}.tar.gz results/*
mv results_${pid}.tar.gz ../..

cd ../..
rm -rf $CODENAME
