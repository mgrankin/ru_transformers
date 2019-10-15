mkdir .install 
mkdir .conda
cd .install 
wget `wget -O - https://www.anaconda.com/distribution/ 2>/dev/null | sed -ne 's@.*\(https:\/\/repo\.anaconda\.com\/archive\/Anaconda3-.*-Linux-x86_64\.sh\)\">64-Bit (x86) Installer.*@\1@p'` -O anaconda.sh
chmod +x anaconda.sh
./anaconda.sh -b -p $HOME/.anaconda
cd
sed -i '1ieval "$($HOME/.anaconda/bin/conda shell.bash hook)"' .bashrc
source .bashrc

git clone https://github.com/mgrankin/ru_transformers
cd ru_transformers
conda env create -f environment.yml

cd
rm train_setup.sh