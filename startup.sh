#!/bin/bash


echo "===== STARTUP.SH STARTED ====="

apt-get update
apt-get install -y graphviz

echo "===== GRAPHVIZ CHECK ====="
which dot
dot -V

echo "===== STARTING STREAMLIT ====="

streamlit run HomeMain.py --server.port 8000 --server.address 0.0.0.0
