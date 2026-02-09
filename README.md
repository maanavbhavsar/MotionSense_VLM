# MotionSense VLM

A Vision-Language Model (VLM) that combines CLIP-based semantic understanding with motion-aware features for video action recognition.

## Overview

MotionSense VLM enhances video action recognition by fusing:
- **Semantic features**: CLIP-based embeddings from sparse video frames
- **Motion features**: Optical flow statistics extracted from video sequences

This hybrid approach leverages both semantic understanding and temporal motion patterns for improved action classification.

## Project Structure

```
MotionSense_VLM/
├── config/          # Configuration files
├── data/           # Data loading and processing
├── models/         # Model architectures
├── eval/           # Evaluation and benchmarking
├── demos/          # Visualization and demos
├── scripts/        # Main execution scripts
└── outputs/        # Generated outputs
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MotionSense_VLM
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

See `requirements.txt` for the complete list of dependencies.

Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- open-clip-torch (for CLIP models)
- opencv-python >= 4.8.0
- numpy >= 1.24.0

## Usage

*Coming soon - project is in early development phase*

## Development Status

**Current Phase**: Setup & Infrastructure ✅

- [x] Project structure
- [x] Git repository setup
- [x] Dependencies configuration
- [ ] Data loading pipeline
- [ ] Model implementations
- [ ] Training scripts
- [ ] Evaluation tools

## License

*To be determined*

## Contributing

*Guidelines coming soon*
