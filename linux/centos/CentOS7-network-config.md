
```
nmcli connection show
```

```
nmcli connection modify ens7f0 \
connection.autoconnect yes \
ipv4.method manual \
ipv4.address 172.16.66.243/24 \
ipv4.gateway 172.16.66.254 \
ipv4.dns 114.114.114.114
```

```
nmcli connection modify ens5f1 \
connection.autoconnect yes \
ipv4.method manual \
ipv4.address 172.16.66.242/24 \
ipv4.gateway 172.16.66.254 \
ipv4.dns 114.114.114.114
```
