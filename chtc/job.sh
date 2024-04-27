#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

CODENAME=PROPS
export PATH

cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz -C .
rm ${CODENAME}.tar.gz # remove code tarball
cd $CODENAME

# install code
ENVNAME=ppo-ros
ENVDIR=${ENVNAME}
cp /staging/ncorrado/${ENVNAME}.tar.gz .
mkdir $ENVDIR
tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
rm ${ENVNAME}.tar.gz # remove env tarball
source $ENVDIR/bin/activate

pip install -e .
pip install PROPS/custom_envs
cd PROPS

pid=${1}
step=${2}
command_fragment=`tr '*' ' ' <<< $3`
echo $command

python3 -u ${command_fragment} --run-id ${step} --seed ${step}
#$($command --seed $step --run-id $step)
#$($command)

tar -czvf results_${pid}.tar.gz results/*
mv results_${pid}.tar.gz ../..

cd ../..
rm -rf $CODENAME
