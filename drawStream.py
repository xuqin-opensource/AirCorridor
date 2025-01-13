import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import geometry
import NavierStokes
import random

def mapScope(gdf, bbox):
    # 通过边界框过滤数据
    return gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

def mapElevationRange(gdf, elevation):
    return gdf[(gdf['Elevation'] >= elevation)]

def meshing(gdf, minx, miny, maxx, maxy, N=100):
    grid_width = (maxx - minx) / N
    grid_height = (maxy - miny) / N

    # 生成网格
    grid_cells = []
    for i in range(N):
        for j in range(N):
            cell_minx = minx + i * grid_width
            cell_maxx = cell_minx + grid_width
            cell_miny = miny + j * grid_height
            cell_maxy = cell_miny + grid_height
            grid_cells.append(geometry.box(cell_minx, cell_miny, cell_maxx, cell_maxy))

    # 创建一个 GeoDataFrame 存储网络
    grid = gpd.GeoDataFrame({'geometry':grid_cells}, crs=gdf.crs)

    # 计算每个网格单元与建筑数据的交集
    # intersection 会将网格与建筑交集的部分保留下来
    intersection = gpd.overlay(grid, gdf, how='intersection')

    # 计算每个网格的统计信息，比如最高高程
    # 高程数据在'Elevation'字段中
    # grid.geometry.apply 的作用是对 grid GeoDataFrame 中的 geometry 字段（即网格单元的几何对象）逐个进行操作和计算。
    grid['max_elevation'] = grid.geometry.apply(lambda cell: intersection.loc[intersection.geometry.within(cell), 'Elevation'].max())
    # grid['max_elevation'] = grid['max_elevation'].fillna(0)
    return grid

def drawGrid(grid):
    # 可视化网格和统计信息
    grid.plot(column='max_elevation', cmap='viridis', legend=True, figsize=(10, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('上海建筑高程网格化数据', fontsize=16)
    plt.xlabel('经度', fontsize=12)
    plt.ylabel('纬度', fontsize=12)
    plt.show()

def drawStream(u, v):
    seek_points = np.array([[0 for i in range(N)], 
                        [i for i in range(N)]])

    # Plot the results
    plt.figure(figsize=(11, 7), dpi=100)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.contourf(X, Y, u, alpha=0.5, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, u, cmap=plt.cm.viridis)
    plt.scatter(np.array(obstacles)[:, 1], np.array(obstacles)[:, 0], s=1)  # 障碍物
    plt.streamplot(X, Y, u, v, color='k', minlength=0.5, density=1)
    # plt.streamplot(X, Y, u, v, color='r', integration_direction='forward', minlength=1)
    # plt.streamplot(X, Y, u, v, color='b', integration_direction='backward', minlength=1)


    # plt.xlim(0, 100)
    # plt.ylim(0, 100)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'N = {N}, dt={dt}, nu = {nu}')
    plt.show()

def getStreamlines(X, Y, u, v):
    stream = plt.streamplot(X, Y, u, v, color='k', minlength=0.5, density=1)
    return stream.lines.get_segments()

def draw3D(streamlines, grid, grid_width, grid_height):
    # 准备绘制三维柱状图的数据
    # 将网格的 X, Y 坐标作为基础，Z 坐标为高程
    x_pos = []
    y_pos = []
    z_pos = np.zeros_like(grid['max_elevation'])

    dx = np.ones_like(grid['max_elevation']) * grid_width
    dy = np.ones_like(grid['max_elevation']) * grid_height
    dz = grid['max_elevation'].values  # 高程值

    # 填充 x_pos 和 y_pos 数组
    for i, cell in enumerate(grid['geometry']):
        coords = list(cell.exterior.coords)
        x_pos.append((coords[0][0] + coords[2][0]) / 2)  # 网格中心的 X 坐标
        y_pos.append((coords[1][1] + coords[3][1]) / 2)  # 网格中心的 Y 坐标

    # 转换为 numpy 数组
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    # 获取坐标的最小值和最大值（用于坐标对齐）
    x_min, x_max = np.min(x_pos), np.max(x_pos)
    y_min, y_max = np.min(y_pos), np.max(y_pos)

    # 创建三维图
    plt.ioff() # 关闭交互模式（提高渲染速度）
    fig = plt.figure(figsize=(12, 8), dpi=100)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维柱状图
    color_map = plt.get_cmap('Blues')
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, shade=True, color=color_map(dz / max(dz)), alpha=0.3)
    # ax.view_init(elev=20, azim=20)

    # 将流线插入到三维图中
    for streamline in streamlines:
        
        # 随机生成RGB颜色值，每个分量取值范围是0到1
        r = random.random()
        g = random.random()
        b = random.random()
        color = (r, g, b)

        for line in streamlines[streamline]:
            y, x = line[:, 0]*((x_max-x_min)/N), line[:, 1]*((y_max-y_min)/N)

            z = np.zeros_like(x)

            # 在三维图上绘制流线
            ax.plot(x+x_min, y+y_min, z+streamline, color=color, linewidth=2.5, alpha=1.0)

    # 添加标题和标签
    ax.set_title('上海建筑高程数据的三维柱状图', fontsize=16)
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_zlabel('高程')
    plt.show()

if __name__ == "__main__":
    layers = 5; minElevation = 20; maxElevation = 100
    N = 100; nt = 100; nit = 1000; dt = 0.00001; rho = 1; nu = 0.0

    file_name = "上海\Shanghai_Buildings_DWG-Polygon.shp"
    # 读取Shapefile文件
    gdf = gpd.read_file(file_name)

    # 定义一个边界框 [min_lon, min_lat, max_lon, max_lat] 来限制区域
    bbox = [500000, 3450000, 501000, 3451000]  # 根据实际区域修改这些经纬度
    gdf_scope = mapScope(gdf, bbox)
    gdf_elevation = mapElevationRange(gdf_scope, minElevation) # # 筛选高程大于 20 米的建筑

    # 确定数据的边界
    minx, miny, maxx, maxy = gdf_scope.total_bounds
    grid_width = (maxx - minx) / N
    grid_height = (maxy - miny) / N

    # 网格化
    grid = meshing(gdf_elevation, minx, miny, maxx, maxy, N=N)
    # grid['max_elevation'] = grid['max_elevation'].fillna(0) # 将max_elevation为NaN的数据替换为0

    # drawGrid(grid) # 画出二维平面网格图

    # 网格生成
    nx, ny = N+1, N+1 # 网格的水平和垂直方向的分辨率
    x = np.linspace(0, N, nx)
    y = np.linspace(0, N, ny)
    X, Y = np.meshgrid(x, y)

    streamlines = {} # 字典类型存储每一层流线，key为流线高度，value为流线值
    for i in range(layers):
        elevation = minElevation + (maxElevation - minElevation) / layers * i
        theta = 90 / layers * i

        # 建筑物数据
        obstacles = [[int(i / N), i % N] for i in range(len(grid)) if grid['max_elevation'][i]>=elevation]
        
        u, v, p = NavierStokes.navier_stokes(N=N, obstacles=obstacles, nx=nx, ny=ny, theta=theta, nt=nt, nit=nit, dt=dt, rho=rho, nu=nu)

        streamlines[elevation] = getStreamlines(X, Y, u, v)
    
    grid = grid.dropna(subset=['max_elevation']) # 删去为Nan的数据行

    draw3D(streamlines=streamlines, grid=grid, grid_width=grid_width, grid_height=grid_height)