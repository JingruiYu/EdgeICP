# Edge ICP

<p align="center">
  <img src="https://img.shields.io/badge/language-C++-blue" />
  <img src="https://img.shields.io/badge/topic-ICP-orange" />
  <img src="https://img.shields.io/badge/topic-edge--alignment-lightgrey" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" />
</p>

[中文](#中文说明) | [English](#english)

## Contents

- [Overview](#english)
- [Repository structure](#repository-structure)
- [Related publication](#related-publication)
- [Citation](#citation)
- [License](#license)
- [中文说明](#中文说明)


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

## Related publication

This repository is an experimental extension inspired by the following paper:

- [**ViLiVO: Virtual LiDAR-Visual Odometry for an Autonomous Vehicle with a Multi-Camera System**](https://ieeexplore.ieee.org/document/8968484/)

It is **not** the original implementation of ViLiVO. The repository focuses on edge / contour based pose-estimation experiments that extend related geometric-localization ideas.

## Project status

This is an experimental repository. It may require dependency and dataset-path adaptation before being used in a modern environment.


## Citation

If you use this repository, please cite or acknowledge it using the metadata in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the [Apache License 2.0](LICENSE). Please retain the license and notice files when redistributing or reusing the code.

---

## 中文说明

这是一个基于 **C++** 的边缘 / 轮廓匹配实验项目，主要探索如何利用 **ICP 风格的迭代优化** 做位姿估计。

项目关注道路边界、车道线、物体轮廓、BEV 边缘等几何结构在视觉定位中的作用，适合作为边缘约束、轮廓匹配、几何定位实验的参考。

## 关键词

ICP、边缘匹配、轮廓匹配、位姿估计、视觉定位、BEV、自动驾驶、OpenCV、PCL、C++。

## 相关论文

该仓库是受以下论文启发的实验性延伸：

- [**ViLiVO: Virtual LiDAR-Visual Odometry for an Autonomous Vehicle with a Multi-Camera System**](https://ieeexplore.ieee.org/document/8968484/)

需要注意：该仓库**不是** ViLiVO 的原始实现，而是围绕边缘 / 轮廓约束位姿估计做的延伸性实验。

## 引用与许可

如果你使用该仓库，请通过 [`CITATION.cff`](CITATION.cff) 引用或致谢该项目。许可协议见 [`LICENSE`](LICENSE)。
