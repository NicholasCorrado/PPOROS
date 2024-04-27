#!/bin/bash

# set up conda env
ENVNAME=ppo-ros
ENVDIR=${ENVNAME}_env

cp /staging/ncorrado/${ENVNAME}_env.tar.gz .
mkdir $ENVDIR
tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
rm ${ENVNAME}.tar.gz # remove env tarball
source $ENVDIR/bin/activate
