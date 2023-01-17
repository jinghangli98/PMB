sudo umount /Volumes/storinator
sudo mkdir /Volumes/storinator

sudo sshfs -o allow_other,defer_permissions,IdentityFile=~/.ssh/id_rsa jil202@136.142.190.89:/home /Volumes/storinator
