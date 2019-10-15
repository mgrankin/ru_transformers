provider "google" {
 credentials = "${file(".gcp_credentials.json")}"
 project     = "gpt2train"
 region      = "us-west1"
}

resource "google_compute_address" "ip_address" {
  name = "my-address"
}

// A single Google Cloud Engine instance
resource "google_compute_instance" "default" {
 name         = "train-instance"
 machine_type = "n1-standard-1"
 zone         = "us-west1-a"

 boot_disk {
   initialize_params {
     image = "ubuntu-1804-lts"
     size = 20
   }
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
