provider "google" {
 credentials = "${file("../.gcp_credentials.json")}"
 project     = "gpt2train"
 region      = "us-west1"
}

resource "google_compute_address" "ip_address" {
  name = "my-address"
}

resource "google_compute_disk" "data-disk" {
  name  = "data-disk"
  type  = "pd-ssd"
  size = 200
  zone  = "us-west1-a"
}

// A single Google Cloud Engine instance
resource "google_compute_instance" "default" {
  name         = "train-instance"
  machine_type = "n1-standard-32"
  zone         = "us-west1-a"

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

data "google_tpu_tensorflow_versions" "available" { }

resource "google_tpu_node" "tpu" {
    name               = "test-tpu"
    zone               = "us-central1-b"

    accelerator_type   = "v2-8"

    cidr_block         = "10.3.0.0/29"
    tensorflow_version = "${data.google_tpu_tensorflow_versions.available.versions[0]}"

    description = "Terraform Google Provider test TPU"
    network = "default"

    labels = {
        foo = "bar"
    }

    scheduling_config {
        preemptible = true
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
    ports    = ["22", "80", "8080", "1000-2000"]
  }

  target_tags = ["train"]
}

resource "google_compute_network" "default" {
  name = "open-network"
}

output "instance_ips" {
  value = ["${google_compute_address.ip_address.address}"]
}
