# KENSHIN 待办清单

> 记录待办事项。阶段进度与 [docs/staged-plan.md](docs/staged-plan.md) 同步维护；
> 临时事项、想法、优化点记在下面"杂项待办"里，做完打勾或删除。

---

## 阶段进度（与 staged-plan.md §5 同步）

**Phase 0 · 工程与基础设施**
- [x] S0.1 仓库与双项目构建系统（engine/app）✅ 2026-08-31
- [x] S0.2 spdlog 日志 / Assert / Timer / Catch2 接入 ✅ 2026-08-31

**Phase 1 · 窗口与 Vulkan 底座**
- [x] S1.1 Win32 窗口与输入 ✅ 2026-08-31
- [x] S1.2 Vulkan 1.4 设备初始化（volk+VMA+特性报告，shaderBufferInt64Atomics 即停检查）✅ 2026-08-31
- [x] S1.3 交换链 + 帧循环 + 截图管线 ✅ 2026-08-31
- [x] S1.4 Slang 着色器管理（.slang→SPIR-V，uint64 原子冒烟）✅ 2026-09-01
- [x] S1.5 ImGui + 相机 + 帧统计面板 ✅ 2026-09-01

**Phase 2 · 基线渲染路径 → M0 验收**
- [ ] S2.1 MeshData / OBJ 加载 / 程序化网格
- [x] S2.2 基线前向渲染路径 ✅ 2026-09-01
- [ ] S2.3 光照 + swapchain 重建 + 千帧长跑

**Phase 3 · 离线 Builder（meshopt+clusterlod）→ M1a/M1 验收**
- [x] S3.1 meshopt+clusterlod 集成冒烟 ✅ 2026-09-01
- [ ] S3.2 DAG 装配（父子边/bounds/单调误差传播）
- [x] S3.3 .kgeo 序列化 + kgeo_builder CLI ✅ 2026-09-01（M1a）
- [ ] S3.4 LOD 切割面可视化验证（M1，无裂缝检查）

**Phase 4 · 虚拟几何运行时 → M2 验收**
- [x] S4.1 GPU Scene 上传 ✅ 2026-09-02
- [x] S4.2 实例剔除 CS + CPU 差分 ✅ 2026-09-02
- [x] S4.3 簇剔除 CS（LOD 决策）+ 差分 ✅ 2026-09-02
- [x] S4.4 软光栅 CS（uint64 VisBuffer）+ 逐像素差分 ✅ 2026-09-02
- [x] S4.5 材质解码 pass（对照基线双路径截图）✅ 2026-09-02（`--kgeo X --virtual [--checker]`；交互崩溃三连修：① Slang 模块缓存 use-after-free（ShaderManager 现持有模块引用）② 软光栅半像素行偏移（修后 98.2% 逐字节一致、裂缝 10px 与基线持平）③ 近距缩放 DEVICE_LOST——驱动对 payload 依赖寻址链/u64 位算术喂地址的着色器模式会硬崩，decode 改 u32 读 vis + 全索引钳制 + 数值净化规避；顺带修 S4.4 双重 view 变换、decode vis 行序、剔除朝向、shaderInt64/shaderDrawParameters 特性、VMA_DYNAMIC_VULKAN_FUNCTIONS 显式关闭）
- [x] S4.6 调试视图 + 阈值滑条 + 计数曲线（M2）✅ 2026-09-02（四视图 + `--cam`/`--view`/`--multi` CLI + Virtual Geometry 面板；验收数据见 M2 条目）

## 里程碑验收

- [x] **M0**（Phase 2 后）：窗口+Vulkan+ImGui+相机+基线网格渲染稳定 ✅ 2026-09-01
- [x] **M1a**（S3.3 后）：builder 全链路 + 单测全绿 ✅ 2026-09-01
- [x] **M1**（S3.4 后）：LOD 各层级渲染无裂缝 ✅ 2026-09-01
- [x] **M2**（S4.6 后）：端到端虚拟几何画面，三角形数由分辨率决定 ✅ 2026-09-02（2026-09-03 修正实例包围球缺失后复测：三实例 `--multi` 阈值单调 T=0.25/1/4 → 68250/39321/29100 三角严格单调；分辨率 1280×720→2560×1440 → 像素写入 52289→209213=4.00×（恰为像素比）、切割三角 45752→58836=1.29× 亚线性自适应；LOD 视图近深远浅渐变清晰；校验层全程零错误；验收机位改 `--cam 270,25,14,10,0,0` 三实例全入画）

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

## Vulkan RHI v2 重构

（详见 docs/vulkan-rhi-plan.md，2026-09-01 调研定稿）

- [ ] P0（M2 前置）：拆 CommandContext/DeletionQueue/DescriptorCache/UploadEngine + 每帧单一 rendering scope + staging ring
- [ ] P1（M4 前置）：timeline semaphore + compute 队列 + GPU timer
- [ ] P2（M6 前置）：分页上传/传输队列/驻留预算

## 杂项待办

（随手记，格式：`- [ ] 事项 —— 备注/日期`）

- [ ] CommandContext 记 last-bound 状态（pipeline/顶点/索引缓冲），`draw()` 跳过重复绑定——当前每帧 draw 数个位数、驱动对冗余绑定有优化，纯属顺手加固；改动只在 backend 内部，不动 API —— 2026-09-02
- [ ] one-shot 链入帧化（当前视角变化触发阻塞式 5 段 one-shot compute,交互帧率受限;运动中 VisBuffer 落后相机一帧导致剪影处闪烁/破洞,停止即恢复——链入帧+VisBuffer 按帧槽双缓冲可同时解决卡顿与错位;ShaderManager 的 SPIR-V 缓存已就位为前置）—— S4.5 落地后,M2 后续
- [ ] 软光栅差分盲区对策：GPU/CPU 差分两边共享实现缺陷时不可见（S4.4 半像素行偏移、双重 view 两个 bug 都是 S4.5 对 HW 基线对比才暴露的）；后续可加一条 float64 独立地面真值 raster 校验 —— 2026-09-02
- [ ] （示例）调研 clusterlod.h 的 clodConfig 各参数含义 —— 对应 S3.1
- [ ] RenderView 抽象（多视口/多窗口时引入；Filament View=Scene+Camera、renderEngine RenderView 同位）—— 2026-09-01 Renderer/IRenderPipeline 落地时明确推迟
- [ ] RenderGraph（pass 图 + 资源生命周期 + 调度；参照 Frostbite FrameGraph / UE RDG / Godot RDG；UI 合成届时从 backend endFrame 上移为真 pass）—— M2 多 pass 化时引入
- [ ] Input 迁入子系统容器（当前 static 单例被 Window 的 WndProc 直接喂数据，解耦需动消息泵）—— 2026-09-01 子系统容器落地时评估
- [x] VulkanBackend：per-draw rendering scope 合并为每帧单一 scope（CommandContext::beginViewportPass/endPass，2026-09-01 P0-2 完成并字节级验证）
