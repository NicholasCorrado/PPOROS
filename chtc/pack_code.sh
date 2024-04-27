cd ../.. # cd just outside the repo
tar --exclude='./PROPS/chtc/results' --exclude='./PROPS/results' -czf epymarl.tar.gz epymarl
# cd ..
cp PROPS.tar.gz /staging/ncorrado