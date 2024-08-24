# SFM: Structure From Motion

This repository provides a simple Structure-from-Motion (SFM) pipeline.

## Installation

```
pip install -r requirements.txt
```

## Demo

To run the demo and see the pipeline in action:

```
python demo.py
```

## Sample Output

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
    <div style="flex: 0 0 auto; margin-right: 10px;">
        <p>1. Feature Extraction</p>
        <img src="image.png" alt="Feature Extraction" height="200"/>
    </div>
    <div style="flex: 0 0 auto; margin-right: 10px;">
        <p>2. Feature Matching</p>
        <img src="image-1.png" alt="Feature Matching" height="200"/>
    </div>
    <div style="flex: 0 0 auto;">
        <p>3. Sparse Reconstruction</p>
        <img src="image-2.png" alt="Sparse Reconstruction" height="200"/>
    </div>
</div>
