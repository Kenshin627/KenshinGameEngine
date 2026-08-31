# KEngine 待办清单

> 记录待办事项。阶段进度与 [docs/staged-plan.md](docs/staged-plan.md) 同步维护；
> 临时事项、想法、优化点记在下面"杂项待办"里，做完打勾或删除。

---

## 阶段进度（与 staged-plan.md §5 同步）

**Phase 0 · 工程与基础设施**
- [x] S0.1 仓库与双项目构建系统（engine/app）✅ 2026-08-31
- [x] S0.2 spdlog 日志 / Assert / Timer / Catch2 接入 ✅ 2026-08-31

**Phase 1 · 窗口与 Vulkan 底座**
- [ ] S1.1 Win32 窗口与输入
- [ ] S1.2 Vulkan 1.2 设备初始化（volk+VMA+特性报告，shaderBufferInt64Atomics 即停检查）
- [ ] S1.3 交换链 + 帧循环 + 截图管线
- [ ] S1.4 Slang 着色器管理（.slang→SPIR-V，uint64 原子冒烟）
- [ ] S1.5 ImGui + 相机 + 帧统计面板

**Phase 2 · 基线渲染路径 → M0 验收**
- [ ] S2.1 MeshData / OBJ 加载 / 程序化网格
- [ ] S2.2 基线前向渲染路径
- [ ] S2.3 光照 + swapchain 重建 + 千帧长跑

**Phase 3 · 离线 Builder（meshopt+clusterlod）→ M1a/M1 验收**
- [ ] S3.1 meshopt+clusterlod 集成冒烟
- [ ] S3.2 DAG 装配（父子边/bounds/单调误差传播）
- [ ] S3.3 .kgeo 序列化 + kgeo_builder CLI（M1a）
- [ ] S3.4 LOD 切割面可视化验证（M1，无裂缝检查）

**Phase 4 · 虚拟几何运行时 → M2 验收**
- [ ] S4.1 GPU Scene 上传
- [ ] S4.2 实例剔除 CS + CPU 差分
- [ ] S4.3 簇剔除 CS（LOD 决策）+ 差分
- [ ] S4.4 软光栅 CS（uint64 VisBuffer）+ 逐像素差分
- [ ] S4.5 材质解码 pass（对照基线双路径截图）
- [ ] S4.6 调试视图 + 阈值滑条 + 计数曲线（M2）

## 里程碑验收

- [ ] **M0**（Phase 2 后）：窗口+Vulkan+ImGui+相机+基线网格渲染稳定
- [ ] **M1a**（S3.3 后）：builder 全链路 + 单测全绿
- [ ] **M1**（S3.4 后）：LOD 各层级渲染无裂缝
- [ ] **M2**（S4.6 后）：端到端虚拟几何画面，三角形数由分辨率决定

---

## 后续里程碑（排期未定，选做）

- [ ] **ECS 实体组件系统**（gameplay 层，用户确认将来要做：引入 EnTT（header-only，vcpkg）；渲染器保持只消费"实例平表"的接口缝，ECS 在上游填数组，管线零改动）
- [ ] M3 两遍 HZB 遮挡剔除
- [ ] M4 大三角形 HW 光栅路径 + async compute 重叠
- [ ] M5 BVH8 + Persistent Threads 剔除
- [ ] M6 分页流式加载
- [ ] M7 量化压缩（八面体法线/隐式切线/拓扑编码）
- [ ] M8 Imposter / VSM / TAA
- [ ] Interfaces/ DLL 公共 API 边界层（参照 renderEngine，需要分发时）
- [ ] TextureManager（需要纹理时）

## 杂项待办

（随手记，格式：`- [ ] 事项 —— 备注/日期`）

- [ ] （示例）调研 clusterlod.h 的 clodConfig 各参数含义 —— 对应 S3.1
