cd ../.. # cd just outside the repo
tar --exclude="PROPS/plotting" --exclude="chtc" -czvf PROPS.tar.gz PROPS
cp PROPS.tar.gz /staging/ncorrado
