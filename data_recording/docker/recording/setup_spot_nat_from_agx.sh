#!/bin/bash

NUC_USER="cvg-nuc-1"
NUC_IP="192.168.1.243"
ORIN_IFACE="enp88s0"
SPOT_IFACE="enxa0cec8e56d7c"

echo "[INFO] Enabling Spot NAT via SSH on NUC..."

ssh -t ${NUC_USER}@${NUC_IP} << EOF
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o ${SPOT_IFACE} -j MASQUERADE
sudo iptables -A FORWARD -i ${SPOT_IFACE} -o ${ORIN_IFACE} -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A FORWARD -i ${ORIN_IFACE} -o ${SPOT_IFACE} -j ACCEPT
echo "NAT forwarding rules set on NUC."
EOF
