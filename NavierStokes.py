import numpy as np

# 泊松方程源项的生成
def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                           ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                           2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                           ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

# 泊松方程的求解
def pressure_poisson(p, dx, dy, b, nit):
    pn = np.empty_like(p)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:- 1])

        # Boundary conditions for pressure
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0
    return p

# 纳维-斯托克斯方程的求解
def navier_stokes(N, obstacles, nx, ny, theta, nt=100, nit=1000, dt = 0.00001, rho=1, nu=0.0):
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)

    # 定义旋转角度
    theta = np.radians(theta)  # 将角度转为弧度

    # 初始化场变量
    u = np.ones((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    # 应用旋转矩阵
    u_rot = u * np.cos(theta) - v * np.sin(theta)
    v_rot = u * np.sin(theta) + v * np.cos(theta)

    # 更新速度场
    u = u_rot
    v = v_rot

    # np.empty_like返回一个新数组，其形状和类型与给定数组相同。
    # 这个函数不会初始化数组中的数据，因此它包含的是任意值，这可能使得它比初始化数组值的函数（如np.zeros_like或np.ones_like）稍微快一点。
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        build_up_b(b, rho, dt, un, vn, dx, dy)
        p = pressure_poisson(p, dx, dy, b, nit)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Apply boundary conditions
        # u[0, :] = 1  # Inlet
        # u[-1, :] = 1  # Outlet
        # u[:, 0] = 1  # Inlet
        # u[:, -1] = 1  # Outlet
        # v[0, :] = 0
        # v[-1, :] = 0
        # v[:, 0] = 0
        # v[:, -1] = 0

        # boundary conditions
        
        # 8个网格
        # for i, j in obstacles:
        #     u[i, j], v[i, j] = 0, 0
        #     for dx, dy in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
        #         # 检查边界条件
        #         if 0 <= i+dx < N and 0 <= j+dy < N:
        #             u[i+dx, j+dy], v[i+dx, j+dy] = 0, 0

        # 四个网格
        for i, j in obstacles:
            u[i, j], v[i, j] = 0, 0
            for dr, dc in [[-1, 0], [0, -1], [0, 1], [1, 0]]:
                # 检查边界条件
                if 0 <= i+dr < N and 0 <= j+dc < N:
                    u[i+dr, j+dc], v[i+dr, j+dc] = 0, 0

    return u, v, p