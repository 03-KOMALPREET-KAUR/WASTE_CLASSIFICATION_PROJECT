---
app_port: 7860
title: AI Waste Classification
emoji: 🗑️
colorFrom: green
colorTo: blue
sdk: static
python_version: 3.11 # Crucial fix for stability and compatibility
run: |
  pip install -r requirements.txt
  gunicorn --workers 1 --timeout 120 app:app -b 0.0.0.0:7860
---

**AI-Powered Waste Classification and Recycling Suggestions** ♻️

**Project Overview**

This project is a deep learning-based image classification system that can identify waste as biodegradable or non-biodegradable, and further classify it into categories such as plastic, metal, paper, cardboard, organic waste, clothes, shoes, batteries, and glass.

It also provides guidance on reusability, recyclability, and recommends appropriate disposal methods, supporting sustainable waste management and environmental protection.

**Key Features:**

✅ Binary Classification: Biodegradable vs Non-Biodegradable

✅ Multiclass Classification: Plastic, Metal, Paper, Cardboard, Organic, Clothes, Shoes, Batteries, Glass

✅ Reusability & Recycling Suggestions

✅ High Accuracy: EfficientNetB0 model with 92.97% validation accuracy

✅ User-friendly Web Interface

✅ Scalable for smart dustbins, mobile apps, and municipal systems
