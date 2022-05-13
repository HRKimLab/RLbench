# make mount point if not exists
sudo mkdir -p /nfs/share
# mount nfs (nfs server need to allow this computer)
sudo mount 10.201.185.92:/volume1/neurlab-ann /nfs/share/
