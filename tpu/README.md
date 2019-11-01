# WIP
### 1. Initialize credentials
```gcloud init```
### 2. Create a project dedicated to train a NN
```bash
gcloud projects create gpt2train
gcloud config set project gpt2train
```

Go to the web interface and link billing account to your project. I don't have a script for that.

### 3. Attach gcloud to Terraform
```bash
gcloud iam service-accounts create terraform
gcloud iam service-accounts keys create ./.gcp_credentials.json \
  --iam-account terraform@gpt2train.iam.gserviceaccount.com
gcloud config set project gpt2train
gcloud services enable cloudbilling.googleapis.com
gcloud services enable compute.googleapis.com

gcloud projects add-iam-policy-binding gpt2train \
  --member serviceAccount:terraform@gpt2train.iam.gserviceaccount.com \
  --role roles/editor

gcloud iam service-accounts get-iam-policy \
    terraform@gpt2train.iam.gserviceaccount.com

```
### 4. Create instance with Terraform

```bash
cd 00_prepare/
terraform init
terraform plan
terraform apply
```

### 5. Setup an instancce

```bash
IP=34.70.206.131 # your node IP
scp train_setup.sh ubuntu@$IP:
# dataset packed with 'tar -caf data.zst.tar data/'. To unpack tar -xaf
rsync -vP data.zst.tar ubuntu@$IP:  

# go there 
ssh ubuntu@$IP 

sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
bash ./train_setup.sh

# docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
sudo apt install docker-ce -y
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo reboot

```

### 6. Create an image for preemptive instance

```bash
gcloud compute images create train-image --source-disk train-instance --source-disk-zone us-central1-b --force
#gcloud compute images delete train-image 
```

### 7. Replace instance with Terraform

```bash
cp 00_prepare/terraform.tfstate 01_train/
cd 01_train/

terraform plan
terraform apply
```

### 8. Run learning

I'm trying to use transfer learning here. The vocab is different to the original, so at first I freeze all the layers but the embeddings and the last linear layer. After it stops improoving I unfreeze next layers (one attention layer from start and one from the end) and decrease the LR. The parameter `--unfreeze_level` tells how much to unfreeze. The rule of thumb is - perplexity on larger model should be lower than perplexity on smaller model at the end of each unfreezing step. 

```bash
IP=34.70.206.131 # your node IP
ssh ubuntu@$IP 

tmux new -s a # to recover in case of disconnect
# I need xm.save() function, it's only in xla:nightly right now
docker pull gcr.io/tpu-pytorch/xla:nightly
docker run -v /home/ubuntu/ru_transformers:/root/ru_transformers --expose	6006 -it --shm-size 60G gcr.io/tpu-pytorch/xla:nightly 

# inside docker container
cd
cd ru_transformers
git pull 
pip install -r tpu_requirements.txt

export TPU_IP_ADDRESS=10.3.0.2 # this ip may change, it's yours tpu ip
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export XLA_USE_BF16=1 
export TRAIN_FILE=./data/full

# test if it's working at all
python /pytorch/xla/test/test_train_mp_mnist.py

# choose the size and run your training

# GPT-2 124M
export MODEL_SIZE=gpt2
export OUTPUT=output/full_s
export BS=8
export LR=40e-4

# GPT-2 355M
export MODEL_SIZE=gpt2-medium
export OUTPUT=output/full_m
export BS=4
export LR=24e-4

# GPT-2 774M
export MODEL_SIZE=gpt2-large
export OUTPUT=output/full_l
export BS=1
export LR=8e-4

# Repeat with UNFREEZE=1,2,3,...

export UNFREEZE=0
export NUM_EPOCH=30.0

./fit.sh


# if TPU hangs - del/new with this commands
terraform destroy -target=google_tpu_node.tpu -auto-approve
terraform apply -target=google_tpu_node.tpu -auto-approve

# to watch tensorboard
docker ps
docker exec -it 00a0b14a7675 /bin/bash
cd; cd ru_transformers/output/classic_s/
tensorboard --logdir runs --host 0.0.0.0

```

### 9. Results

Your perplexity will be different, depending on the tokenizer, the vocab and the dataset. The better your tokenizer the worse your perplexity, actually.

Perplexity on the validation set:

GPT-2                           | Small, 124M  | Medium, 355M   | Large, 774M | 
---                                  | -- | ---                          | --- | 
Unfreeze 0, BS=64, LR 40e-4, 60 epoch        | Train loss 5.41, Eval PP 121 |                           |   | 
Unfreeze 0, BS=64, LR 5e-4, 20 epoch         | Train loss 5.37, Eval PP 120 |                           |   | 

Unfreeze 1, BS=64, LR 40e-4, 60 epoch         | Train loss 4.6, Eval PP 61 |                           |   | 
checkpoint-230000
Unfreeze 1, BS=64, LR 5e-4, 20 epoch         | Train loss 4.54, Eval PP 59.72 |                           |   |  
checkpoint-256416

