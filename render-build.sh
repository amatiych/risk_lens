#!/usr/bin/env bash
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Build React frontend
cd web
npm ci
npm run build
cd ..
