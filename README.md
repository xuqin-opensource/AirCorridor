# AirCorridor
# Redefining Metropolitan Skies: Fluid Dynamics-Driven Air Corridors for Urban Air Mobility
本项目基于Navier-Stokes方程实现城市空域走廊的生成。

## Key Features
## 主要特性
- Supports Shapefile format for urban building data
- 支持Shapefile格式的城市建筑数据导入
- 
- Multi-layer wind field simulation (5 layers by default)
- 多海拔分层风场模拟（默认5层）
- 
- Grid-based processing of building elevation data
- 基于网格化的建筑高程数据处理
- 
- 3D streamline visualization
- 三维流线可视化
- 
- Adjustable fluid dynamics parameters
- 可调节的流体动力学参数

## Environment Requirements
- Python 3.7+
- Required libraries:
  ```bash
  pip install geopandas matplotlib numpy shapely

## Modify parameters in main.py (optional):
### Basic parameters
- layers = 5        # Number of simulation layers
- minElevation = 20 # Minimum elevation (meters)
- maxElevation = 100# Maximum elevation (meters)
- N = 100           # Grid resolution

### Fluid parameters
- nt = 100          # Time steps
- nit = 1000        # Pressure correction iterations for Poisson equation
- dt = 0.00001      # Time step size
- rho = 1           # Fluid density
- nu = 0.0          # Kinematic viscosity

### Region bounding box
- bbox = [500000, 3450000, 501000, 3451000]  # Adjust according to actual coordinates

# Data
- Place Shanghai building data Shapefiles in the 上海/ directory
- Ensure the following files are present:
- - Shanghai_Buildings_DWG-Polygon.shp
- - Shanghai_Buildings_DWG-Polygon.shx
- - Shanghai_Buildings_DWG-Polygon.dbf
- - Shanghai_Buildings_DWG-Polygon.prj

- The Shanghai building data used in the research is sourced from https://www.bilibili.com/opus/670165423935717413.
- The copyright of the original data belongs to JackLloydSmith.
