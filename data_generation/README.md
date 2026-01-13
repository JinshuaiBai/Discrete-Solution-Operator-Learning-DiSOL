# DiSOL Geometry Generator (MATLAB)

This folder provides a minimal MATLAB pipeline to generate **2D random geometries** and **random contiguous boundary segments** on a Cartesian grid, consistent with the procedure described in the DiSOL Supplementary Information.

The output can be used to build input channels for downstream PDE dataset generation and DiSOL training.

---

## Files

- `create2DGeom.m`  
  Generates a random **closed 2D geometry** (binary mask) on an `ImgSize × ImgSize` grid.

- `selectBoundary.m`  
  Extracts the **longest boundary** of the geometry mask and randomly samples a **contiguous boundary segment** (with wrap-around).

- `generateData.m`  
  Batch generator that connects `create2DGeom` + `selectBoundary` to produce a 4D tensor:
  - `geom_datasets`: `[N × 2 × H × W]` (uint8)
    - Channel 1: geometry mask (`1 = inside geometry`)
    - Channel 2: boundary segment mask (`1 = selected boundary pixels`)

- `main.m`  
  Example entry script to configure parameters, run batch generation, and save outputs.

---

## Requirements

- MATLAB R2018b+ recommended.
- Uses MATLAB built-ins:
  - `boundary`, `bwboundaries`, `inpolygon`
- Uses Spline Toolbox functions:
  - `augknt`, `spmak`, `fnval`

---

## Quick Start

1. Open MATLAB and set the working directory to this folder.
2. Run:

```matlab
main
