# WIP
### Initialize credentials
```gcloud init```
### Create a project dedicated to train a NN
```
TF_VAR_billing_account=
gcloud projects create gpt2train
gcloud config set project gpt2train
```

Go to the web interface and link billing account to your project. I don't have a script for that.

### Attach gcloud to Terraform
```
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
### Create instance with Terraform

```
cd 00_prepare/
terraform init
terraform plan
terraform apply
```

### Setup an instance

```
IP=35.185.201.94 # your node IP
scp train_setup.sh ubuntu@$IP:
# dataset packed with 'tar -caf data.zst.tar data/'
rsync -vP data.zst.tar ubuntu@$IP:  

# go there 
ssh ubuntu@$IP 

sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
bash ./train_setup.sh
sudo -s
crontab -l | { cat; echo "@reboot mount /dev/sdb /home/ubuntu/ru_transformers/output"; } | crontab -
exit

```

### Create an image for preemptive instance

```
gcloud compute images create train-image --source-disk train-instance --source-disk-zone us-west1-a --force
#gcloud compute images delete train-image 
```
