
#vast ai template: cuda:12.0.1-devel-ubuntu20.04

sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncurses-dev \
  xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git nano

pip install --upgrade pip


curl https://pyenv.run | bash


echo 'export PYENV_ROOT="$HOME/.pyenv"' >> .profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> .profile
echo 'eval "$(pyenv init - bash)"' >> .profile

pyenv install 3.12
pyenv global 3.12

git clone https://github.com/sile16/turbozero.git
pip install git+https://github.com/sile16/turbozero.git

apt-get -y install libcairo2-dev

# if necessary to avoid XLA errors
export XLA_FLAGS="--xla_gpu_autotune_level=0"