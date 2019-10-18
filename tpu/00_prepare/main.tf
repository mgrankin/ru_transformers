variable region {
  description = "Region for cloud resources."
  default     = "us-central1"
}

variable zone {
  description = "Zone for managed instance groups."
  default     = "us-central1-b"
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
  name         = "train-instance"
  machine_type = "n1-standard-1"
  zone = "${var.zone}"

  boot_disk {
    device_name = "basic-disk"
    initialize_params {
      image = "ubuntu-1804-lts"
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
    ports    = ["22"]
  }

  target_tags = ["train"]
}

output "instance_ips" {
  value = ["${google_compute_address.ip_address.address}"]
}
