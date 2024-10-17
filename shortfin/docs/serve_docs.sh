#!/bin/bash

# Check if dependencies are installed
if ! pip freeze | grep -q -f requirements.txt; then
    echo "Installing dependencies..."
    python3 -m pip install -r requirements.txt
else
    echo "Dependencies already installed."
fi

# Build the documentation
echo "Building documentation..."
sphinx-build -b html . _build

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Documentation built successfully."

    # Start a Python HTTP server
    echo "Starting local server..."
    echo "View your documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server."
    cd _build
    python3 -m http.server 8000
else
    echo "Error: Documentation build failed."
fi
