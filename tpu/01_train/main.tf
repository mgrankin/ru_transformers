variable region {
  description = "Region for cloud resources."
  default     = "us-central1"
}

variable zone {
  description = "Zone for managed instance groups."
  default     = "us-central1-a"
}

provider "google" {
 credentials = "${file("../.gcp_credentials.json")}"
 project     = "gpt2train"
 region      = "${var.region}"
 zone        = "${var.zone}"
}

resource "google_compute_address" "ip_address" {
  name    = "my-address"
  region  = "${var.region}"
}

resource "google_compute_disk" "data-disk" {
  name  = "data-disk"
  type  = "pd-ssd"
  size = 200
  zone = "${var.zone}"
}

resource "google_compute_network" "default" {
  name = "open-network"
}


// A single Google Cloud Engine instance
resource "google_compute_instance" "default" {
  allow_stopping_for_update = true
  name         = "train-instance"
  machine_type = "n1-highmem-32"
  #machine_type = "n2-highmem-48"
  zone = "${var.zone}"

  boot_disk {
    device_name = "basic-disk"
    initialize_params {
      image = "train-image"
      type = "pd-ssd"
      size = 200
    }
  }

  attached_disk {
    source = "${google_compute_disk.data-disk.name}"
  }

  metadata_startup_script = "sudo apt update; sudo apt upgrade -y"

  tags = ["train"]

  metadata = {
    ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
  }

  network_interface {
    network = "open-network"

    access_config {
        nat_ip = "${google_compute_address.ip_address.address}"
    }
  }
}

resource "google_compute_firewall" "default" {
  name    = "test-firewall"
  network = "open-network"

  allow {
    protocol = "icmp"
  }

  allow {
    protocol = "tcp"
    ports    = ["22", "6006-6008"]
  }

  allow {
    protocol = "udp"
    ports    = ["60000-61000"]
  }
  target_tags = ["train"]
}

output "instance_ips" {
  value = ["${google_compute_address.ip_address.address}"]
}

resource "google_tpu_node" "tpu" {
    name               = "train-instance"
    zone               = "${var.zone}"

    accelerator_type   = "v3-8"

    cidr_block         = "10.3.0.0/29"
    tensorflow_version =  "pytorch-nightly" 

    description = "ru_transformers TPU"
    network = "open-network"

    // TFRC is awesome
    //scheduling_config {
    //    preemptible = true
    //}
    
}


data "google_tpu_tensorflow_versions" "available" { }

output "test" {
  value = ["${data.google_tpu_tensorflow_versions.available}", "${google_tpu_node.tpu.network_endpoints}"]
}


resource "google_tpu_node" "tpu2" {
    name               = "train-instance-medium"
    zone               = "${var.zone}"

    accelerator_type   = "v3-8"

    cidr_block         = "10.3.1.0/29"
    tensorflow_version =  "pytorch-nightly" 

    description = "ru_transformers TPU"
    network = "open-network"
}

resource "google_tpu_node" "tpu3" {
    name               = "train-instance-large"
    zone               = "${var.zone}"

    accelerator_type   = "v3-8"

    cidr_block         = "10.3.2.0/29"
    tensorflow_version =  "pytorch-nightly" 

    description = "ru_transformers TPU"
    network = "open-network"
}


/**/