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
terraform init
terraform plan
terraform apply
```

### Setup an instance

```
IP=35.185.201.94 # your node IP
scp train_setup.sh ubuntu@$IP:
ssh ubuntu@$IP bash ./train_setup.sh
```

