# KENSHIN 虚拟几何系统 —— 分阶段（Phase/Stage）实施与测试方案

> 配套文档：《KENSHIN 虚拟几何系统（类 Nanite）从零搭建方案》（总体架构，已批准）。
> 本文档把总体方案拆成 **5 个 Phase、20 个 Stage**。核心原则：**每个 Stage 独立实现、独立验证，
> 验证不通过不进入下一 Stage；每个 Phase 收尾有里程碑验收**。这样任何 bug 都被限制在单个 Stage
> 的小范围内、且在引入当下就被抓住，不会堆积到管线成型后才发现。
>
> - **Phase 0** 工程与基础设施（S0.1–S0.2）
> - **Phase 1** 窗口与 Vulkan 底座（S1.1–S1.5）
> - **Phase 2** 基线渲染路径（S2.1–S2.3）→ **里程碑 M0 验收**
> - **Phase 3** 离线 Builder：构建算法用 meshoptimizer+clusterlod，装配/序列化/验证自己写（S3.1–S3.4）→ **里程碑 M1a/M1 验收**
> - **Phase 4** 虚拟几何运行时（S4.1–S4.6）→ **里程碑 M2 验收**

---

## 0. 总则

### 0.1 三类验证手段（贯穿所有阶段）

| 手段 | 说明 | 用在哪 |
|:--|:--|:--|
| **V1 单元测试** | Catch2（vcpkg），`ctest` 运行；含"已知答案"资产与**CPU 参考实现的差分测试** | 纯算法/数据结构（Builder、数学） |
| **V2 无头运行** | demo 支持 `--frames N --size WxH --screenshot out.bmp` 自动渲染并退出，产出 BMP 截图供人工/自动核验 | 渲染类阶段（每阶段固定视角固定参数，可复现） |
| **V3 日志断言** | 运行时统计（簇数/三角形数/剔除计数）打印结构化日志，脚本可比对 | GPU 管线各 pass |

### 0.2 每阶段完成标准（Definition of Done）

1. `cmake --build --preset <release>` 编译通过（/W4，新增代码零警告）；
2. 该阶段验证全部通过（V1/V2/V3 按需）；
3. 阶段产物 git commit 一次（信息格式 `[feat]: S3.2 DAG assembly (metadata layer) + unit tests`，全英文）。

### 0.3 差分测试原则（本项目的特色防线）

GPU 侧每个 pass（剔除、光栅化）都写一个**CPU 参考实现**（同算法的串行版本，放在 tests/ 中），
固定视点下 GPU 结果读回与 CPU 参考逐项比对。这是验证"并行版没写错"的最硬手段，
也是 Nanite 这类并行管线排错的核心方法。

### 0.4 已知答案测试资产（tests/assets/ 内置，全部已知顶点/面数/拓扑统计）

| 资产 | 已知答案 | 用于 |
|:--|:--|:--|
| `tetra.obj` 四面体 | 4 顶点 / 4 三角形 / 6 边，全部边被 2 个三角形共享，0 边界边，Euler V−E+F=2 | 邻接/边界检测 |
| `cube.obj` 立方体 | 8 顶点 / 12 三角形 / 18 边（6 对角+12 棱），0 边界边，V−E+F=2 | 邻接、顶点去重 |
| `grid.obj` 8×8 平面网格 | 81 顶点 / 128 三角形 / 32 条边界边，全部共面 | 簇化/简化行为基线（共面网格 LOD 误差应≈0） |
| `two_sheets.obj` 两个不相连的平面片 | 2 个连通分量 | 图划分孤岛兜底、多 DAG 根 |
| 程序化环面结 | 参数化生成，可设定精确三角形数（如 1,048,576） | 规模/性能/尺度无关性 |
| 环面结 ×100 缩放副本 | 与 ×1 副本簇数/层级数结构一致，误差呈 100× 比例 | 简化误差绝对性（ErrorAbsolute）与运行时 LOD 阈值行为验证 |

### 0.5 全局约定（防经典 bug，一次定死）

- **坐标系**：右手系、Y-up、单位=米；**正面 = 逆时针（CCW）绕序**；SW 光栅绕序剔除、投影矩阵、OBJ 加载（OBJ 默认 CCW）全按此约定，CPU/GPU 参考实现共用 `kenshin/math/` 同一套公式与常量。
- **深度**：Vulkan [0,1]，**反向 Z 不用**（v1 直接 z 线性/非线性均可，但 VisBuffer 的 depth 量化公式 CPU/GPU 必须同一份代码导出的常量）。
- **demo 场景定义**（S3.4/S4.6 的截图验证依赖它有层次）：同一网格（环面结）摆 **近（1m）/中（8m）/远（60m）三组实例** + 一个 cube 参照物，固定 `--cam` 参数集（近景/远景两张）；LOD 层级视图必须呈现"近深远浅"的渐变才算过。

---

## 1. 阶段总览（按 Phase 分组）

### Phase 0 · 工程与基础设施 —— 产出：可构建的双项目仓库 + 日志/测试底座

| 阶段 | 内容 | 验证手段 |
|:--|:--|:--|
| S0.1 | 仓库与双项目构建系统（engine/app） | 构建+运行空壳+ctest |
| S0.2 | spdlog 日志接入 / Assert / Timer / Catch2 接入 | V1 冒烟测试 |

### Phase 1 · 窗口与 Vulkan 底座 —— 产出：能呈现画面、能编 shader、有调试 UI 的空引擎

| 阶段 | 内容 | 验证手段 |
|:--|:--|:--|
| S1.1 | Win32 窗口 + 输入事件 | V3 日志 |
| S1.2 | Vulkan 1.4 设备初始化（volk+VMA+特性报告） | V3 设备报告（含 shaderBufferInt64Atomics 检查，失败即停） |
| S1.3 | 交换链 + 帧循环 + 清屏呈现 + **截图管线** | V2 截图（确定性：同参数逐字节一致） |
| S1.4 | Slang 着色器管理（.slang→SPIR-V 运行时编译） | V1+V3（**uint64 原子 compute 冒烟** + 错误日志带行号） |
| S1.5 | ImGui + 相机 + 帧统计面板 | V2 截图 + 相机矩阵断言 |

### Phase 2 · 基线渲染路径 —— 产出：传统管线画网格（后续一切验证的对照组）→ **M0 验收**

| 阶段 | 内容 | 验证手段 |
|:--|:--|:--|
| S2.1 | MeshData / OBJ 加载 / 程序化网格生成 | V1（已知答案资产） |
| S2.2 | 基线前向渲染（上传+管线+深度+绘制） | V2 截图 + CPU 投影参考比对 |
| S2.3 | 光照 + resize 重建 swapchain + 千帧长跑 | V2 截图 + V3 |

### Phase 3 · 离线 Builder（meshoptimizer + clusterlod.h 承担构建算法）—— 产出：.kgeo（cluster DAG）→ **M1a/M1 验收**

| 阶段 | 内容 | 验证手段 |
|:--|:--|:--|
| S3.1 | meshopt+clusterlod 集成（簇化/分组/锁边简化/递归循环全用库） | V1（尺寸上限/覆盖/层级数/误差）+ V3 构建日志 |
| S3.2 | DAG 装配（父子边/bounds/组决策数据/单调误差传播，手写元数据层） | V1（无环/单调/同组一致/父子覆盖） |
| S3.3 | .kgeo 序列化 + kgeo_builder CLI | V1 roundtrip + V3 百万面构建报告 → **M1a** |
| S3.4 | LOD 切割面可视化验证（基线路径渲染） | V2 截图（**无裂缝**检查）→ **M1** |

### Phase 4 · 虚拟几何运行时 —— 产出：完整 Nanite 式画面 → **M2 验收**

| 阶段 | 内容 | 验证手段 |
|:--|:--|:--|
| S4.1 | GPU Scene 上传（簇表/顶点池/索引池 SSBO） | V1（GPU 读回 CRC32 比对） |
| S4.2 | 实例剔除 CS | V1 **差分**（CPU 参考逐实例比对） |
| S4.3 | 簇剔除 CS（LOD 决策） | V1 差分 + 切割有效性结构断言 |
| S4.4 | 软光栅 CS（uint64 VisBuffer） | V1 **逐像素差分光栅**（核心防线） |
| S4.5 | 材质解码 pass（重心/插值/解析导数） | V2 双路径截图对比（vs 基线） |
| S4.6 | 调试视图 + 阈值滑条 + 计数曲线 | V2 截图组（0.25/1/4px 阈值）→ **M2** |

---

## 2. 阶段详述

### 阶段 S0.1 —— 仓库与双项目构建系统

**目标**：仓库结构落定，"engine 项目 + app 项目"两个 CMake project 单元可构建、可运行空壳。
**目录结构参照 C:\code\refactor\renderEngine 的组织模式**（include/<engine>/ + src/ 镜像、backend 为唯一裸 GPU API 调用处、raii/manager/renderer/scene 模块分目录、Engine 门面 + ISubsystem 子系统容器）。

**改动**：
```
KENSHIN/
├── CMakeLists.txt            # 工作区：add_subdirectory(engine app tools tests)
├── CMakePresets.json         # windows-msvc-debug / -release，vcpkg toolchain (C:\vcpkg)
├── vcpkg.json                # volk vma imgui glm catch2 spdlog meshoptimizer slang
├── .gitignore                # build/ .vs/ *.user out/ log/ 等
├── engine/                   # ── 引擎项目 project(KENSHIN) → kenshin_engine 静态库 ──
│   ├── CMakeLists.txt
│   ├── include/kenshin/           # 内部公共头（include/ke 与 src/ 镜像，参照 renderEngine）
│   │   ├── Common.h          # KENSHIN_NS_BEGIN/END、基础类型、版本
│   │   ├── Engine.h          # 引擎门面 + 子系统容器（type-indexed map）
│   │   ├── ISubsystem.h      # 子系统接口：onInit/onShutdown/onBeginFrame/onEndFrame
│   │   ├── core/             # Window(Win32)、Timer、Input、Version
│   │   ├── backend/          # RenderBackend.h 抽象 + vulkan/（唯一裸 Vulkan 调用处）
│   │   ├── raii/             # GPU 资源 RAII：Buffer/Image/ShaderModule/Pipeline/DescriptorSet/Sync/Swapchain
│   │   ├── manager/          # ShaderManager(Slang 运行时编译)（TextureManager 后续）
│   │   ├── renderer/         # 管线层：BaselineRenderer / VirtualGeometryRenderer / GpuTimer / 各 Cull/Raster pass
│   │   ├── scene/            # Camera / CameraController / RenderObject(实例) / RenderView
│   │   ├── geometry/         # ★ 虚拟几何 builder：ClusterLodBuilder / ClusterDag / ClusterDagBuilder / KGeoFile
│   │   ├── assets/           # MeshData / ObjLoader / ProceduralMesh
│   │   ├── math/             # Bounds(AABB/球) / Frustum / ScreenError / Ray
│   │   ├── util/             # BmpWriter / Crc32
│   │   └── DebugOverlay.h    # ImGui 调试面板宿主（对应 renderEngine::DebugOverlay）
│   ├── src/                  # 与 include/ke 镜像的实现目录
│   └── shaders/              # Slang 源（.slang，运行时编译到 SPIR-V）
├── app/                      # ── 应用项目 project(KENSHINApp) → KENSHINDemo ──
│   ├── CMakeLists.txt        # 链接 kenshin_engine
│   └── src/main.cpp          # 打印版本日志后退出码 0（后续长成 demo 场景）
├── tools/CMakeLists.txt      # 预留（S3.3 填 kgeo_builder CLI）
└── tests/CMakeLists.txt      # 预留（S0.2 填 Catch2）
```

注：renderEngine 的 `Interfaces/` 公共 API 边界层（DLL 分发、app 只见接口）暂不引入——KENSHIN 的 app 在同仓库静态链接，直接 include 引擎头；将来要 DLL 分发时再加该层（结构已为此留位）。命名风格对齐参考项目：PascalCase 类名、m_ 成员前缀、I 前缀接口、KENSHIN_NS 宏。

**验证**：
```bash
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
./build/release/app/KENSHINDemo.exe    # 打印 "KENSHIN 0.0.1 (engine=<sha>, build=<cfg>)" 退出码 0
ctest --preset windows-msvc-release    # 空测试集，通过
```
**通过标准**：以上四条全部成功；`git log` 有初始 commit。

---

### 阶段 S0.2 —— 基础设施：spdlog 日志 / Assert / Timer / 测试框架

**目标**：集成 **spdlog**（vcpkg）并做轻封装：`KENSHIN_INFO/WARN/ERROR` 宏、彩色控制台 + 滚动文件（`log/ke.log`，5MB×3 轮转）、模块 tag 分类；断言宏（Release 可关）；高精度计时器（QueryPerformanceCounter）；Catch2 接入并有第一个冒烟测试。

**改动**：`engine/include/kenshin/core/{Log.h,Assert.h,Timer.h}` + `engine/src/core/{Log.cpp,Timer.cpp}`（Log 为 spdlog 薄封装，fmt 随 spdlog 附带）；`tests/src/smoke.cpp`（断言 `1+1==2`、日志字符串格式化）。

**验证**：ctest 绿（1 test）；demo 日志文件生成且含级别标签。
**通过标准**：V1 通过；日志同时落控制台与文件。

---

### 阶段 S1.1 —— Win32 窗口与输入

**目标**：`kenshin::core::Window`（Win32）：创建/标题/尺寸、WM_SIZE/WM_CLOSE/键盘/鼠标事件 → 回调；`kenshin::engine::Input` 维护按键/鼠标增量状态。无任何 GPU。

**改动**：`include/kenshin/core/Window.h`、`src/core/Window.cpp`、`include/kenshin/core/Input.h` 等；app 主循环轮询事件并打印 resize/按键日志。

**验证**：V3——运行 demo：日志出现 `Window created 1600x900`；手动拖拽窗口，日志出现 resize 序列；ESC 打印并退出（本轮无需截图）。
**通过标准**：事件无丢失（按键按下/抬起成对）、resize 不崩溃。

---

### 阶段 S1.2 —— Vulkan 1.4 设备初始化

**目标**：volk 加载 → instance（校验层按需开启+debug messenger）→ 物理设备选择（独显优先，打印名称/API 版本）→ 逻辑设备（Vulkan 1.4，开启 `shaderBufferInt64Atomics`、`timelineSemaphore`，校验不支持则记录并允许继续）→ 图形+计算队列 → VMA 初始化。

**改动**：`include/kenshin/backend/RenderBackend.h`（首批抽象类型与枚举）+ `src/backend/vulkan/{VulkanContext,VulkanDevice}.cpp`（volk/VMA/设备/队列，参照 renderEngine "backend 是唯一裸 GPU API 调用处"的纪律）；app 启动时创建并打印**设备能力报告**，随后销毁退出。

**验证**：V3——日志含：
```
[Vulkan] instance 1.2 ok, validation=off
[Vulkan] GPU: NVIDIA GeForce RTX 5060 Ti (api 1.4.x, type=discrete)
[Vulkan] feature shaderBufferInt64Atomics = supported   ← 关键，VisBuffer 依赖
[Vulkan] queues: graphics=0 compute=0 (same family), present=ok
```
**通过标准**：报告完整且 shaderBufferInt64Atomics=supported（RTX 5060 Ti 预期支持；不支持则停下与用户确认回退策略）。

---

### 阶段 S1.3 —— 交换链 + 帧循环 + 截图管线

**目标**：surface（VK_KHR_win32_surface）→ swapchain（3 缓冲 FIFO）→ 帧同步（per-frame 信号量+围栏）→ begin/endFrame → 清屏色呈现（颜色随帧号变化）。**同时落地 V2 无头验证管线**：`--frames N --screenshot path --size WxH --cam yaw,pitch,dist[,target]`（相机参数显式指定 ⇒ 所有 V2 截图可复现）；渲染走**离屏 color image**（TRANSFER_SRC 布局）→ blit 到 swapchain 呈现，截图从离屏 image 读取（不碰 swapchain 的 present 布局），vkCmdCopyImage→host 缓冲→写 BMP（手写 BMP 编码器 ~80 行，零依赖）。

**改动**：`src/backend/vulkan/{Swapchain,FrameSync}.cpp` + `include/kenshin/raii/`（Image/Imageview 包装）；`src/util/BmpWriter.cpp`；app 参数解析。

**验证**：V2——`KENSHINDemo --frames 3 --size 640x360 --screenshot shot.bmp` 退出码 0；BMP 文件 640×360；第 3 帧清屏色与日志中帧号对应的预期色一致（脚本读 BMP 首像素比对）。
**通过标准**：三次连续运行截图逐字节一致（确定性渲染）。

---

> **Vulkan 基线 = 1.4**（用户 2026-08-31 定；两块本机 GPU 均支持 1.4.x）：shaderBufferInt64Atomics 自 1.2 起核心仍需显式开启；**1.3 起 dynamic rendering 免 render pass/framebuffer 样板、synchronization2 屏障 API**，**1.4 起 push descriptor 免描述符池**——S1.3 起的管线代码按这些现代基线写。

### 阶段 S1.4 —— Slang 着色器管理

**目标**：`ShaderManager`：从 `engine/shaders/` 读 **.slang** 源（选 Slang 而非 GLSL 的理由：语法为 HLSL 超集，Nanite 参考资料里的 shader 几乎可直接搬；官方 SPIR-V 后端支持 64 位整数原子），经 **Slang 运行时库**（slang C API；vcpkg 有 slang port 则用之，否则 vendor 官方二进制发行版到 ThirdParty/slang）编译为 SPIR-V，内存缓存+编译错误带行号日志；支持 `#include` 模块。

**改动**：`include/kenshin/manager/ShaderManager.h` 等；`shaders/test_atomic.slang`——**冒烟即验证核心路径**：`RWStructuredBuffer<uint64_t>` 上 `atomicMax` + 4 元素 reduce，读回结果核对（同时覆盖 Slang 的 uint64 原子→SPIR-V Int64Atomics 能力链路，VisBuffer 的命门）。

**验证**：V1+V3——demo 运行 test_atomic.slang 并读回结果 `[1,2,3,4]→reduce=10`、原子 max=预期值，日志确认；故意写错 shader 时日志带 `error: line N` 且程序不崩溃。
**通过标准**：uint64 atomicMax 路径编译执行正确（若 HLSL 风格 `InterlockedMax` 方法语法在 SPIR-V 后端有坑（issue #4120），改用 Slang 原生 `atomicMax()` 自由函数写法；两者都不通则回退 GLSL+shaderc，编译器接口双支持成本很低）。

---

### 阶段 S1.5 —— ImGui 集成 + 相机

**目标**：ImGui（Vulkan 后端）接入帧循环；`Camera`（透视投影+视图矩阵，orbit：LMB 旋转/滚轮缩放，WASD+QE 平移）；ImGui 窗口显示：FPS 曲线、帧时间 ms、相机位置、引擎信息。投影矩阵做 Vulkan 适配（Y 翻转 + 深度 [0,1]，GLM_FORCE_DEPTH_ZERO_TO_ONE）。

**改动**：`src/scene/{Camera,CameraController}.cpp`、`src/DebugOverlay.cpp`（ImGui 宿主）；app 每帧更新。

**验证**：V2——`--screenshot` 出现 ImGui 面板（文字与曲线可见）；相机受输入影响（脚本注入固定键盘事件序列后，视图矩阵日志与 CPU 预期一致）。
**通过标准**：面板正常渲染、不闪烁（双缓冲描述符正确）；相机矩阵正交性检查通过。

---

### 阶段 S2.1 —— 几何资产层

**目标**：`MeshData`（SOA：pos f32×3 / normal f32×3 / uv f32×2 + u32 索引）；手写 OBJ 加载（v/vn/vt/f，多边形扇三角化，缺属性补默认，索引重映射；MTL：Ka/Kd/texture 名，v1 单材质）；程序化生成器（UV 球、环面结，可选 fBm 位移，参数直接控制三角形数）。

**改动**：`include/kenshin/assets/{MeshData.h,ObjLoader.h,ProceduralMesh.h}` + 实现；tests：加载 `tetra/cube/grid` 与已知答案比对（计数与包围盒精确匹配）；程序化生成器固定参数 → 固定三角形数。

**验证**：V1 全绿。
**通过标准**：三种 OBJ 计数/包围盒精确匹配；环面结 1,048,576 三角形生成 < 2s。

---

### 阶段 S2.2 —— 基线前向渲染路径

**目标**：GPU buffer 上传（staging copy）、图形管线（顶点布局 PNU、push constant MVP、深度测试）、绘制网格。这是后续所有验证的**对照基线**。

**改动**：`include/kenshin/renderer/BaselineRenderer.h` + `src/renderer/BaselineRenderer.cpp`；`shaders/baseline.slang`(VS/FS entry points)。相机矩阵走 **UBO**（3×mat4=192B 超 push constant 保证下限 128B，push constant 只放小参数）。

**验证**：V2——固定相机画 cube 与环面结（不位移），截图与手工推导的投影包围盒比对（cube 八顶点应落在预期屏幕坐标 ±2px）；`tests` 中加 CPU 投影参考。
**通过标准**：深度正确（近面遮挡远面，截图像素检查两个已知点颜色）；三角形绕序正确（无背面错误）。

---

### 阶段 S2.3 —— 光照与 swapchain 重建

**目标**：Blinn-Phong（单方向光+环境，法线来自顶点插值）、背景渐变；窗口 resize → 等待 in-flight → 重建 swapchain/framebuffer → 恢复渲染。

**验证**：V2——光照截图（上亮下暗的球/结）；V3——resize 事件后继续渲染 N 帧无错误。
**通过标准**：**M0 验收达成**：窗口+Vulkan+ImGui+相机+基线网格渲染稳定（含 --frames 1000 长跑无泄漏无崩溃）。

---

### 阶段 S3.1 —— meshoptimizer + clusterlod.h 集成（构建冒烟）

**目标**：引入 meshoptimizer（vcpkg，版本须 ≥ v0.25/1.0，提供 `buildMeshletsSpatial`/`partitionClusters`/带 `Sparse|ErrorAbsolute|LockBorder` flag 的 `simplify`）与其官方单头微库 **clusterlod.h**（demo 目录，MIT；固定 commit 落到 `ThirdParty/clusterlod/`，vcpkg 版本过旧则整库 vendor）——后者就是"簇化→分组→锁边→简化→重簇化→递归→DAG"完整构建循环的生产级实现（zeux 2025 文章的主角）。本阶段先跑通：对测试网格执行 `clodBuild`，把每层每组的簇结构（三角形数、顶点数、LOD 误差）打进日志。

**改动**：`include/kenshin/geometry/ClusterLodBuilder.h`（对 clusterlod.h 的封装：MeshData → clodMesh 适配）+ 实现；tests：
- 尺寸上限：每簇三角形 ≤128 **且顶点 ≤256**（256 = 光栅器 groupshared 缓存上限）；
- 覆盖：第 0 层簇的三角形并集 = 输入网格全部三角形，无重复；
- 层级数 ≈ log₂(N/128)±2（grid 128 三角形 → 1-2 层；1M 三角形环面结 → ~13±2 层）；
- 误差：每簇误差有限、非负、随层级递增；
- **属性保留（关键关卡）**：带 UV+法线输入，验证简化后的簇顶点仍带合理属性（UV 无跳变、法线与几何一致）——若 clusterlod.h 不支持属性传递，**降级方案：改用 meshopt 原语（buildMeshlets→partitionClusters→simplifyWithAttributes）自组构建循环**（循环逻辑手写 ~200 行，算法仍全部用库），并记录进风险表。

**验证**：V1 全绿 + V3 构建日志（层数/簇数/每组簇数分布/误差范围）。
**通过标准**：`clodBuild` 在全部已知答案资产上跑通且断言通过；环面结 100 万面构建 < 60s（clusterlod 已做多线程优化空间，v1 单线程即可）。

---

### 阶段 S3.2 —— DAG 装配（我方元数据层）

**目标**：把 `clodBuild` 回调产出装配成引擎自己的 DAG 结构（这一层必须自己写，且是运行时剔除/LOD 决策的数据源）：
- **父子边**：每层每组回调中，组内新簇认该组全部旧簇为父（多父边 ⇒ DAG）；
- **簇 bounds**：由簇顶点计算 AABB→包围球；
- **组级决策数据**：同组簇共享（union err, union bounds）——保证同组簇并行 LOD 决策天然一致（same input ⇒ same output）；
- **单调误差传播**：自叶向根 `pErr(c) = max(err(c), 子群 max err)`，bounds 同步取并集（唯一切割的保证）；
- 簇顶点/索引导出为连续池（供 GPU 上传与 .kgeo 序列化）。

**改动**：`include/kenshin/geometry/ClusterDag.h`（DAG 数据结构）+ `ClusterDagBuilder.cpp`（装配 + 传播）；tests：
- **无环**：多父 DAG 的 Kahn 拓扑检查；
- **误差单调**：所有父子边 `pErr(parent) ≥ err(parent)`，且沿根→叶路径父误差 ≥ 子误差；
- **同组一致**：同组成员的 (pErr, bounds) 完全相同；
- **父子覆盖**：父簇三角形数 ≈ 子组并集的 50%±10%（简化率）；
- **多根合法**：two_sheets 产出 ≥1 个根、各分量独立成 DAG 子树。

**验证**：V1 全绿 + V3 构建报告（层数/簇数/根数/最大误差/传播前后误差变化数）。
**通过标准**：全部结构断言通过——这是"用库不等于不管正确性"的关卡，meshopt 产出的结构在装进我们的 DAG 后仍满足 Nanite 语义。

---

### 阶段 S3.3 —— .kgeo 格式与 kgeo_builder CLI

**目标**：二进制格式（little-endian）：`header{magic 'KGE1', version, meshCount} + 簇表{boundsSphere×2f4, err, pErr, childBegin/Count, vertOffset, triOffset} + 顶点池 + 索引池`，全部 16 字节对齐段；`kgeo_builder input.obj -o out.kgeo --report`；引擎侧 Reader。

**改动**：`include/kenshin/geometry/KGeoFile.h` + 读写实现；`tools/kgeo_builder/main.cpp`；tests：roundtrip（写→读→逐字段相等）；tools 加到 ctest（CLI 冒烟：构建 grid.kgeo）。

**验证**：V1 roundtrip 绿；V3——对 1,048,576 三角形环面结跑 CLI：日志输出构建各阶段耗时、层数、簇数、文件大小；文件可重新加载。
**通过标准**：**M1a 验收**——builder 全链路 + 全部单测绿。

---

### 阶段 S3.4 —— LOD 切割面可视化验证（关键正确性关卡）

**目标**：demo 加载 .kgeo，**用基线渲染路径**渲染任意指定切割面：CPU 按屏幕误差规则（同 GPU 公式）选簇，把选中簇的三角形临时拼成 mesh 走 S2.2 管线；视图模式：正常着色 / **LOD 层级着色**（每深度一层颜色）/ 误差热图；LOD 阈值滑条。

**改动**：`src/renderer/LodPreviewRenderer.cpp`（引擎侧）+ app 侧 ImGui 控件。

**验证**：V2 截图组（固定相机，阈值 0.5/2/8px 三档）：
- **无裂缝**：层级着色截图放大检查边界（人工 + 辅助：同一相机下相邻阈值渲染的深度差分统计，裂缝区域深度=背景，比例应为 0）；
- 层级分布：近处层级深（红）、远处层级浅（蓝）渐变合理；
- 计数曲线：阈值↑ → 渲染三角形数单调↓，且 8px 档三角形数 < 原始的 5%。
**通过标准**：**M1 验收**——无裂缝 + 单调性成立（这是 builder 正确性的最终裁判）。

---

### 阶段 S4.1 —— GPU Scene 上传

**目标**：.kgeo → 常驻 GPU 缓冲：`ClusterTable`（bounds/err/pErr/偏移）、`VertexPool`、`IndexPool`、`InstanceData`（transform+meshId）；**多 mesh 聚合**（MeshRegistry：每个 .kgeo 追加进共享池，记录 per-mesh 簇范围 [clusterBegin, clusterEnd) 与实例的 meshId 映射）；描述符集绑定；上传走 staging。

**改动**：`include/kenshin/renderer/VirtualGeometryScene.h` + 实现。

**验证**：V1——上传后读回 GPU 缓冲做 CRC32 与源数据一致；V3 日志（缓冲尺寸/偏移）。
**通过标准**：校验和一致；100 万面网格上传 < 0.5s。

---

### 阶段 S4.2 —— 实例剔除 CS + 差分

**目标**：`cull_instances.slang`：每线程 1 实例，视锥 6 面测试（用 GPU Scene 实例 bounds）→ 原子 append 可见实例列表 + 计数。

**改动**：shader + `src/renderer/InstanceCullPass.cpp`；tests：`cpu_reference_cull.cpp`——CPU 同公式视锥剔除，GPU 读回逐实例比对。

**验证**：V1——8 个实例（含跨近平面、跨左平面、完全在外、完全在内边界 case）GPU/CPU 结果 100% 一致。
**通过标准**：差分零差异。

---

### 阶段 S4.3 —— 簇剔除 CS + 切割面有效性

**目标**：`cull_clusters.slang`：每线程 1 个**工作项 = (可见实例, 簇)**——先由展开 pass（或同 shader 内 upper-bound 循环）把可见实例 × 其 mesh 簇范围展开成工作项表（v1 用 scan/compaction：实例剔除输出 → 线性 scan 算前缀和 → 展开写入 (instanceIdx, clusterIdx) 对；后续里程碑改 persistent 遍历时自然消失）：对每项做实例变换下包围球投影屏幕半径 → 误差像素化 → 决策 `pErrPix > T && errPix ≤ T` → append 可见簇列表（带 instanceIdx）+ 写 indirect dispatch 参数 + 统计原子计数。

**改动**：shader + Pass；tests：CPU 参考（同一批固定视点）逐簇比对 + **切割有效性检查**：选出的簇集满足"若选簇 c 则其所有祖先链上恰有一个簇被选"（沿父链断言）且同组簇决策一致。

**验证**：V1 差分 + 结构断言全绿；V3 可见簇数日志（与 S3.4 CPU 选簇数一致）。
**通过标准**：差分零差异、切割结构有效。

---

### 阶段 S4.4 —— 软光栅 CS + 差分光栅（项目核心）

**目标**：`rasterize_sw.slang`：1 工作组 = 1 可见簇、128 线程 = 128 三角形；Phase1 顶点变换（MVP→NDC→视口，groupshared，>128 顶点循环）；Phase2 边方程 setup + 包围矩形（clamp 到视口）+ 绕序剔除（背面面积符号判断）+ **近平面裁剪**（跨近平面三角形按 Sutherland-Hodgman 对 z=near 裁剪，1 个三角形可能裂成 0-2 个，线程内串行处理；完全在近平面外跳过）+ 半空间增量求值（先不做 scanline 优化）；写像素 = `atomicMax(visbuffer[pixel], pack(depth,cluster,tri))`，depth 反向量化高位。

**改动**：shader + `SoftRasterPass.cpp` + VisBuffer 管理（清零用 fill）；tests：`cpu_reference_raster.cpp`——CPU 实现同一光栅化规则（像素中心采样+top-left+同一近平面裁剪），**单簇场景 GPU 读回 VisBuffer 逐像素与 CPU 比对**（深度、cluster、tri 三字段）。

**验证**：
- V1 冒烟：单个超大三角形簇 → 覆盖整屏（统计像素数=预期）；
- V1 差分：3 个固定簇（含部分出屏、共享边相邻、不同深度重叠）→ 逐像素一致；
- V1 近平面：**专门构造跨近平面三角形的测试簇**（部分顶点在近平面前），GPU/CPU 逐像素一致、无"三角形整体消失"；
- V3：三角形写像素计数与理论值一致。
**通过标准**：差分零差异（含像素中心恰在边上的 tie case——CPU/GPU 用同一定点规则）。

---

### 阶段 S4.5 —— 材质解码 pass

**目标**：`decode_material.slang` 全屏三角形：读 VisBuffer → 解码 cluster/instance → 取 3 顶点 → 变换屏幕坐标 → **解析求重心**（2×2 逆）→ 插值 normal/UV → **解析导数**（三角形 UV/位置 delta 推 dUV/dxy，喂 textureLod；跨簇不连续天然安全）→ Blinn-Phong 输出交换链。

**改动**：shader + `MaterialDecodePass.cpp`；UV 棋盘格调试纹理（程序化，验证 UV 与导数）。

**验证**：V2——与 S2.3 基线路径**同机同视角双路径截图对比**：像素差分统计（允许深度/导数量化带来的 <0.5% 像素差异，几何边缘需重合）；UV 棋盘格在接缝两侧 mip 一致（无黑边/噪点）。
**通过标准**：双路径画面视觉一致；UV 接缝无导数爆炸伪影。

---

### 阶段 S4.6 —— 调试视图与统计（M2 收官）

**目标**：ImGui 视图切换（最终着色 / LOD 层级色 / 误差热图 / VisBuffer 解码色）；阈值滑条（0.25–4px）；每帧统计（可见簇、光栅三角形、像素覆盖率）曲线图。

**验证**：V2 截图组：
- 三档阈值下三角形计数曲线**稳定在像素数决定的水平**（同一分辨率不同 LOD 阈值单调；放大分辨率 2× → 三角形数 ~2×）；
- 层级视图近细远粗分布合理；
- 统计无断言/无 GPU 校验层错误。
**通过标准**：**M2 验收**——端到端虚拟几何画面达成，光栅三角形数由屏幕分辨率决定而非场景复杂度。

---

## 3. 风险关卡（各阶段触发的决策点）

| 触发条件 | 动作 |
|:--|:--|
| S1.2 shaderBufferInt64Atomics 不支持 | 停止，与用户确认 32 位打包回退方案 |
| S1.4 Slang uint64 原子→SPIR-V 编译路径有坑（issue #4120：HLSL 风格 InterlockedMax） | 先换 Slang 原生 atomicMax() 写法；仍不通则回退 GLSL+shaderc（ShaderManager 接口双支持成本低） |
| S1.4 vcpkg 无 slang port 或版本过旧 | vendor 官方二进制发行版（slang.dll+头）到 ThirdParty/slang |
| S3.1 clusterlod.h 不保留顶点属性（UV/法线）——它是为纯位置网格设计的 | 降级：用 meshopt 原语自组构建循环（partitionClusters→simplifyWithAttributes），算法仍全用库 |
| S3.1 vcpkg 的 meshopt 版本过旧（缺 partitionClusters 空邻接修复/新 simplify flag） | 整库 vendor 到 ThirdParty/（单一 C 库，体积小），固定 commit |
| S3.2 装配断言失败 | 二分定位：库输出行为（读 clusterlod.h 源码对拍）vs 我方装配 bug（单测隔离修） |
| S3.4 发现裂缝 | 按三类来源定位：锁边失效（检查 clodBuild 输出的边界顶点是否移动）/量化不一致（暂无量化，排除）/跨层共存（S3.2 DAG 断言复查） |
| S4.4 差分不一致且无法收敛 | 暂时降精度浮点→定点对齐，记录差异像素位置聚类分析 |
| 任一阶段性能远超预算 | 记录不阻塞（性能是 M3+ 的优化目标），加 TODO 跟踪 |

## 4. 后续里程碑（本方案不含，达到 M2 后另立方案）

M3 两遍 HZB 遮挡剔除 → M4 大三角形 HW 光栅路径 + async compute 重叠 → M5 BVH8 + Persistent Threads → M6 分页流式 → M7 量化压缩（八面体法线/隐式切线/拓扑编码）→ M8 Imposter / VSM / TAA。

---

## 5. 进度跟踪

**Phase 0 · 工程与基础设施**
- [x] S0.1 仓库与构建系统 ✅ 2026-08-31
- [x] S0.2 基础设施与测试框架 ✅ 2026-08-31

**Phase 1 · 窗口与 Vulkan 底座**
- [x] S1.1 Win32 窗口与输入 ✅ 2026-08-31
- [x] S1.2 Vulkan 设备初始化 ✅ 2026-08-31
- [ ] S1.3 交换链与帧循环与截图管线
- [ ] S1.4 Slang 着色器管理（uint64 原子冒烟）
- [ ] S1.5 ImGui 与相机

**Phase 2 · 基线渲染路径（M0 验收）**
- [ ] S2.1 几何资产层
- [ ] S2.2 基线渲染路径
- [ ] S2.3 光照与 swapchain 重建

**Phase 3 · 离线 Builder（M1a/M1 验收，构建算法用 meshopt）**
- [ ] S3.1 meshopt+clusterlod 集成冒烟
- [ ] S3.2 DAG 装配（元数据层）
- [ ] S3.3 .kgeo 与 CLI（M1a 验收）
- [ ] S3.4 LOD 可视化验证（M1 验收）

**Phase 4 · 虚拟几何运行时（M2 验收）**
- [ ] S4.1 GPU Scene
- [ ] S4.2 实例剔除差分
- [ ] S4.3 簇剔除差分
- [ ] S4.4 软光栅差分
- [ ] S4.5 材质解码对照
- [ ] S4.6 调试视图（M2 验收）
