#!/bin/bash

# List of config files to train
CONFIGS=(
    "config/config_small.yaml"
    "config/config_base.yaml"
    "config/config_large.yaml"
    "config/config_giant.yaml"
)

# Loop through each config file
for CONFIG in "${CONFIGS[@]}"; do
    echo "Training with config: $CONFIG"
    
    # Run the training command
    python train.py --config $CONFIG
    
    # Check if the training succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully trained with config: $CONFIG"
    else
        echo "Failed to train with config: $CONFIG"
        # Optionally, exit the script if one config fails
        # exit 1
    fi
done

echo "All configurations trained successfully!"