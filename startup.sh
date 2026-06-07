#!/bin/bash

apt-get update
apt-get install -y graphviz

streamlit run HomeMain.py --server.port 8000 --server.address 0.0.0.0
