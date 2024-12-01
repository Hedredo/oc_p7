#!/bin/bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"i don't like this, this is bad\"}" \
     -w "\n"

# To run the file, please execute chmod +x request.sh and then ./request.sh