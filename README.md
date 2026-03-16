# PDF Eraser

A web-based tool to select and erase any region in a PDF with AI-powered background reconstruction — plus copy/paste and background cleaning.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-3.1-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Erase** — draw a rectangle over any area to remove it; AI reconstructs the background
- **Copy / Paste** — grab any region, place it anywhere on the page with drag-to-move and resize handles
- **Clean Background** — erase the background behind text while keeping the text and numbers in place
- **AI inpainting** — uses [LaMa](https://github.com/advimman/lama) for complex backgrounds; solid backgrounds get fast color fill
- **Undo / Redo** — button controls and Ctrl+Z / Ctrl+Y keyboard shortcuts
- **Download** — export the edited PDF with all changes baked in

## How It Works

1. Upload a PDF
2. Select a tool: Erase, Clean BG, or Copy
3. Draw a rectangle over the target area
4. For Copy: click Paste, drag to position, resize with handles, then Commit
5. Undo/Redo as needed
6. Download the final PDF

## Setup

```bash
# Clone the repo
git clone https://github.com/nwtsmnt/pdf-eraser.git
cd pdf-eraser

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

The app runs at `http://localhost:5001`.

## Requirements

- Python 3.10+
- System fonts: Nimbus Roman, Nimbus Sans, Nimbus Mono (typically pre-installed on Linux via `gsfonts`)
- CPU-only — no GPU required (though a GPU will speed up LaMa inpainting)

## Tech Stack

- **Backend**: Flask, PyMuPDF, OpenCV, Pillow, NumPy, LaMa inpainting, PyTorch
- **Frontend**: Vanilla JS, pdf.js, Canvas API

## Project Structure

```
pdf-eraser/
├── app.py              # Flask backend (erase, copy/paste, clean BG, inpainting)
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Single-page frontend (editor UI)
└── uploads/            # Temporary uploaded PDFs (gitignored)
```
