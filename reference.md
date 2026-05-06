# Manifold Flow v2.0 实施与测试计划

本项目采用分阶段驱动开发（Phased-Driven Development），确保每一步的数学底层绝对牢固。这种模块化的开发与测试标准不仅能保障后续研究级论文的数据准确性，也完全符合一线大厂对系统架构设计的严格要求。

## 阶段 1：底层核心基建 (Core Foundation)
**目标**：确立数据规范，封死类型漏洞。

* **开发任务**：
  1. 编写 `core/types.py`，配置 NumPy 依赖。
  2. 编写 `core/base_system.py`，实现 `DeterministicSystem` 和 `StochasticSystem` 的抽象类，以及最关键的 `Projectable3D` 接口。
* **测试方案 (Unit Tests)**：
  - 类型检查：测试如果子类不实现 `project_to_3d`，必须在实例化时抛出 `TypeError`。
  - 维度验证：构建一个 Dummy System，断言其内部状态数组形状与投影后 `(3,)` 形状的解耦性。

## 阶段 2：数值积分器实现 (Solvers)
**目标**：分离 ODE 与 SDE，确保数学积分物理意义正确。

* **开发任务**：
  1. 实现 `RK4Solver`。
  2. 实现 `EulerMaruyamaSolver`，引入 `np.random.normal` 生成布朗运动增量。
* **测试方案 (Math Verification)**：
  - **ODE 测试**：使用解析解已知的简单谐振子方程（$y'' = -y$），计算 RK4 在不同步长下的截断误差，验证其具备 $\mathcal{O}(h^4)$ 的收敛精度。
  - **SDE 测试**：使用几何布朗运动 (Geometric Brownian Motion)，模拟 1000 条路径，计算终端状态的均值和方差，与理论解析分布进行 Kolmogrov-Smirnov 检验。

## 阶段 3：经典物理系统迁移 (Classical Systems)
**目标**：打通“系统定义 $\rightarrow$ 求解 $\rightarrow$ 投影”的完整单机 Pipeline。

* **开发任务**：
  1. 将旧版的 `LorenzSystem` 等改写为继承 `DeterministicSystem`。
  2. 去除字符串解析，直接在 `drift` 方法里手写 NumPy 数组运算。
* **测试方案 (Sanity Checks)**：
  - 相空间验证：求解洛伦兹吸引子，断言其是否在标准的蝴蝶形态边界内（如 $z$ 值保持正数区间）。

## 阶段 4：高维机器学习与扩散系统 (ML & Diffusion)
**目标**：真正发挥 v2.0 架构的威力，实现复杂神经网络的收敛过程可视化。

* **开发任务**：
  1. 实现真实的 `CANDYNetwork` 动力学，内部用矩阵乘法计算，`project_to_3d` 提取前三个隐藏层。
  2. 实现 `ScoreBasedDiffusionSystem` (SDE版)，基于真实的线性噪声调度器。
* **测试方案 (Convergence Tests)**：
  - 流形收敛测试：对 CANDY 系统输入高维随机初始分布，验证随时间 $t \to \infty$，投影后的 3D 散点是否收敛至目标流形（如极限环或特定曲面）。
  - 扩散去噪测试：验证前向加噪过程的均值是否收敛至 0，方差是否逼近 1（标准高斯分布）。

## 阶段 5：Web 与数据流重构 (WebSocket Streaming)
**目标**：打通前后端，实现低延迟降维数据传输。

* **开发任务**：
  1. 剥离前端的计算逻辑。
  2. 后端 WebSocket 循环严格执行：`solver.step()` $\to$ `project_to_3d()` $\to$ `socket.emit()`。
* **测试方案 (Integration & Stress Tests)**：
  - 带宽与延迟测试：后端运行 1000 维的复杂模型，验证前端接收到的 payload 依然是轻量级的 3D 坐标，确保 60FPS 渲染不掉帧。