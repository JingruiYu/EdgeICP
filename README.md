# Edge ICP

[中文](#中文说明) | [English](#english)

## English

An experimental C++ project for pose estimation using **edge / contour alignment** and **ICP-style iterative optimization**.

The project explores whether image edges, BEV contours, road boundaries, and other geometric structures can provide useful constraints for localization.

## Motivation

Point-feature matching is not the only way to estimate motion. In driving and robotics scenes, contours such as lane markings, curbs, road boundaries, and object edges can also provide strong geometric information. Edge-based alignment can be useful when texture features are weak or unstable.

## Main ideas

- Extract edge or contour observations from image / BEV data.
- Align observed contours with reference contours.
- Estimate relative pose with ICP-style iterative optimization.
- Use the implementation as a testbed for geometric localization experiments.

## Repository structure

```text
.
├── Data/                     # Historical experiment data
├── ICP.cpp
├── interactive_icp.cpp
├── pcl_ICP.cpp
├── usingPcl.cpp
├── CMakeLists.txt
└── README.md
```

## Keywords

`ICP`, `edge alignment`, `contour matching`, `pose estimation`, `visual localization`, `BEV`, `autonomous driving`, `OpenCV`, `PCL`, `C++`

## Project status

This is an experimental repository. It may require dependency and dataset-path adaptation before being used in a modern environment.

---

## 中文说明

这是一个基于 **C++** 的边缘 / 轮廓匹配实验项目，主要探索如何利用 **ICP 风格的迭代优化** 做位姿估计。

项目关注道路边界、车道线、物体轮廓、BEV 边缘等几何结构在视觉定位中的作用，适合作为边缘约束、轮廓匹配、几何定位实验的参考。

## 关键词

ICP、边缘匹配、轮廓匹配、位姿估计、视觉定位、BEV、自动驾驶、OpenCV、PCL、C++。
