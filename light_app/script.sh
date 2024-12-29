#!/bin/sh
fastapi dev main.py &
streamlit run front.py