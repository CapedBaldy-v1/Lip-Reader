#!/bin/bash
# Setup script for YouTube Dataset Scraper

set -e

echo "=========================================="
echo "YouTube Dataset Scraper Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r youtube_scraper/requirements.txt

# Check if ffmpeg is installed (needed for video conversion)
echo ""
echo "Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg is installed"
else
    echo "⚠ ffmpeg is NOT installed"
    echo ""
    echo "ffmpeg is required for video processing."
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  Fedora: sudo dnf install ffmpeg"
    echo "  macOS: brew install ffmpeg"
fi

# Create data directory
echo ""
echo "Creating data directory..."
mkdir -p youtube_scraper/data

# Check if API key is set
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Get YouTube API Key:"
echo "   - Go to: https://console.cloud.google.com/"
echo "   - Create project → Enable YouTube Data API v3"
echo "   - Create credentials → API key"
echo ""
echo "2. Test the scraper:"
echo "   python youtube_scraper/build_youtube_dataset.py --help"
echo ""
echo "3. Run full pipeline:"
echo "   python youtube_scraper/build_youtube_dataset.py \\"
echo "     --api_key YOUR_API_KEY \\"
echo "     --output_dir /media/agam/Local\\ Disk/the_code/roman/majorproject/youtube_dataset \\"
echo "     --full_pipeline \\"
echo "     --max_per_query 50 \\"
echo "     --require_hd \\"
echo "     --max_workers 4"
echo ""
