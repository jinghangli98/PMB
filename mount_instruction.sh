sudo diskutil umount force /Volumes/storinator
sudo diskutil umount force /Volumes/crc

sudo mkdir /Volumes/storinator
sudo mkdir /Volumes/crc

sudo sshfs -o kill_on_unmount,reconnect,allow_other,defer_permissions,IdentityFile=~/.ssh/id_rsa jil202@136.142.190.89:/home /Volumes/storinator
sudo sshfs -o kill_on_unmount,reconnect,allow_other,defer_permissions,IdentityFile=~/.ssh/id_rsa jil202@h2p.crc.pitt.edu: /Volumes/crc
