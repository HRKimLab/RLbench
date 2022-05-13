# make sure that uid is 1000
if [ $UID -eq 1000 ]
then
	echo "uid is 1000"
else
	echo "uid is $UID. please change uid to 1000"
	exit -1
fi
# make mount point if not exists
sudo mkdir -p /nfs/share
# mount nfs (nfs server need to allow this computer)
sudo mount 10.201.185.92:/volume1/neurlab-ann /nfs/share/
