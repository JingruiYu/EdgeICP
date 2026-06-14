# Edge ICP

An experimental C++ implementation for pose estimation using **edge alignment** and ICP-style optimization.

This repository is part of my earlier visual localization / autonomous-driving perception experiments. It investigates whether edge contours in bird's-eye-view or image-space representations can provide useful geometric constraints for pose estimation.

## Motivation

In driving scenes, lane markings, road boundaries, curbs, and object contours often provide strong geometric cues. Instead of relying only on point features, this project explores edge-based alignment for relative pose estimation.

## Main ideas

- Extract edge / contour observations from image or BEV data.
- Align edge observations with reference contours.
- Estimate pose through ICP-style iterative optimization.
- Use the project as a testbed for geometric localization experiments.

## Technical keywords

- ICP
- Edge alignment
- Pose estimation
- Bird's-eye view
- Visual localization
- Autonomous driving
- C++ / OpenCV

## Repository structure

```text
.
├── Data/                     # Historical experiment data
├── CMakeLists.txt
└── README.md
```

## Notes

This is a historical experimental repository and may not be plug-and-play on a modern environment. I keep it public as a reference for my earlier work in geometric localization and robotics perception.

For my more recent related work, see:

- [pyCuSFM: CUDA-accelerated Structure-from-Motion](https://github.com/nvidia-isaac/pyCuSFM)
- [NVIDIA Isaac Neural Reconstruction](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html)
