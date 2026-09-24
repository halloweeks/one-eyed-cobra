mkdir build
cd build

cmake ..
make -j$(nproc)

cp oec $HOME
cp oec_run $HOME
chmod +x $HOME/oec
chmod +x $HOME/oec_run
cd ../
# $HOME/oec