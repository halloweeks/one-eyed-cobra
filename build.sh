mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)

cp oec_predict  $HOME
cp oec_train    $HOME
cp oec_finetune $HOME
cp oec_img      $HOME

chmod +x $HOME/oec_predict
chmod +x $HOME/oec_train
chmod +x $HOME/oec_finetune

chmod +x $HOME/oec_img