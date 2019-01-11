# Ubuntu 14.04 Install MonoDevelop

## 1. Add the Mono repository to your system

The package repository hosts the packages you need, add it with the following commands.
Note: the packages should work on newer Ubuntu versions too but we only test the ones listed below.

Ubuntu 14.04 (i386, amd64, armhf)

```
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
sudo apt install apt-transport-https
echo "deb https://download.mono-project.com/repo/ubuntu stable-trusty main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
sudo apt update
```

Ubuntu 16.04 (i386, amd64, armhf)

```
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
sudo apt install apt-transport-https
echo "deb https://download.mono-project.com/repo/ubuntu vs-xenial main" | sudo tee /etc/apt/sources.list.d/mono-official-vs.list
sudo apt update
```

Ubuntu 18.04 (i386, amd64, armhf)

```
sudo apt install apt-transport-https dirmngr
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
echo "deb https://download.mono-project.com/repo/ubuntu vs-bionic main" | sudo tee /etc/apt/sources.list.d/mono-official-vs.list
sudo apt update
```

## 2. Install MonoDevelop

```
sudo apt-get install monodevelop
```

The package monodevelop should be installed for the MonoDevelop IDE.

## 3. Verify Installation

After the installation completed successfully, it's a good idea to run through the basic hello world examples on this page to verify MonoDevelop is working correctly.

## 4. References

1. [MonoDevelop Install](https://www.monodevelop.com/download/#fndtn-download-lin)
   
    [https://www.monodevelop.com/download/#fndtn-download-lin](https://www.monodevelop.com/download/#fndtn-download-lin)

