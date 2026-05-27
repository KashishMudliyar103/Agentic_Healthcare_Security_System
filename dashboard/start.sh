#!/bin/bash

# Start API in background
python main.py api &

# Start demo in background (after 5s delay)
sleep 5 && python main.py demo 50 &

# Start dashboard (foreground - this is what Render monitors)
python main.py dashboard