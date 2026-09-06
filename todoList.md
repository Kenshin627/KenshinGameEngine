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
- [x] M3 两遍 HZB 遮挡剔除 ✅ 2026-09-05（M3.1~M3.4 全过，见下方细分；分阶段方案 2026-09-04 批准，编号此前只在会话/代码注释里，2026-09-05 补记于此；明确不做：深度梯度测试、cone backface、实例级遮挡前置（留 M5）、HZB storage image 优化）
  - [x] M3.1 深度金字塔基建（HZB 缓冲式单 dispatch 原子攀爬 + CPU 参考差分）✅ 2026-09-03（commit 2e15436）
  - [x] M3.2 遮挡测试进簇剔除 Main Pass + M3.2b 盒投影紧矩形/链内接线 ✅ 2026-09-05（commit 3eeba91 + 595fa12，详见杂项 M3.2b 条目）
  - [x] M3.3 Post Pass 闭环 ✅ 2026-09-05（被剔簇进 pendingPairs（counters[4] 即计数）；Main 光栅后建本帧初步 HZB → post_cull 复测 pending（只重测遮挡，当前帧 viewProj）→ 复活者补光栅并入同一 VisBuffer → 建最终 HZB 留下帧 Main（每帧两次构建，Nanite 对齐）；HZB 语义改为"帧末建、测上一帧"（1 帧旧替代 2 帧旧），最终 HZB 无条件构建作为激活引导；count_to_indirect 加 cfg[2] 源索引；fillBuffer 补写前屏障（WAR + 跨提交序）；post 光栅/最终 HZB 用独立管线实例（同帧双更新描述符集违例——NVIDIA 绑定时快照碰巧正确，spec 层 UB）；ChainInputs::occlusionEnabled 开关 + 面板 occ-culled/revived 计数与开关。验收：双链差分（轨道相机 7 帧，遮挡开/关）VisBuffer 逐位一致 + pending>0 + revived>0（disocclusion 无洞的强形式）；球/结场景静态相机截图与无遮挡 HEAD 逐字节一致；校验层零错误）
  - [x] M3.4 遮挡密集场景验收 ✅ 2026-09-05（--multi --dense 布局：三实例沿视轴 x=0/2.2/4.4；readStats 补 [11] 切割三角形数——cull 主趟幸存簇 + post 复活簇把 indexCount/3 累进 counters[6]（即退役计划步骤②预告的面板改读链统计）；面板 occ-culled/revived 曲线入列。验收数据：app 640×461 机位 `--cam 270,0,8,2.2,0,0 --lod-threshold 0.25` → 129 簇全切 84 可见 + 45 遮挡剔除 = **35% 剔除**、revived 0（静态相机）、校验层零错误；单测同布局双链差分 VisBuffer 逐位一致 + 削减 ≥25% 断言。**量化记录：默认阈值 1.0 下 0 剔除不是 bug**——粗簇矩形（~90px）超出近球圆盘（r≈85px）的 mip 安全核（mip5=32px tile 时 ≈138px 核直径 vs 93px rect 触到圆盘外 tile），空像素 L0=0（远平面，保守约定——=0xFFFFFFFF 会变"最近遮挡体"错误剔除）混入 min 污染 tileMin → 全保守保留；这正是 M3.2b 记录的"背景污染主导损失"，RECT/mip 收紧是后续优化空间（Nanite 同源保守性））
- [x] M4 大三角形 HW 光栅路径 + async compute 重叠 ✅ 2026-09-05 深夜（M4.1~M4.4 全过，见下方细分；方案 2026-09-05 定稿，**逐项对照 UE5 源码**（知乎《游戏引擎随笔》0x21/0x35/0x36 源码解析 + thecandidstartup）修订；明确不做：scanline 内循环分支（Karis 后测无中间地带）、HW 绑深度目标/合并 pass（Nanite 同弃）、raster bin 材质分箱（无材质系统，UE 5.1 才有）、Vertex Reuse Batch（同上，材质可编程时才需要））
  - [x] M4.1 SW 光栅定点化对齐（正确性前置）✅ 2026-09-05（16.8 子像素定点（SUBPIXEL=256，齐次除后 xy floor——DX 规范的 x.8 量化）+ int64 精确边函数（Slang `long` 不存在用 `int64_t`；shaderInt64 特性早已开启，纯值数学不喂地址无驱动怪癖）+ DX top-left 平局规则（上/左 tie 入、右/下 tie 出；规则在正面积规范绕序下对 yFlip 两种约定自洽——代数验证）；z 保持在量化位置上的屏幕仿射插值（UE MicropolyRasterize 同款，非透视校正）；CPU 镜像同步重写（i64 数学逐算子）。基线重录：SW 差分全过、全量 45 用例绿、校验层零错误、画面目检无裂缝；knot 截图 4.9% 像素变化=平局归属确定化+亚像素量化重写近平局胜者（预期量级）；密集遮挡场景剔除率 25%→20%（量化内缩 ≤1/256px → HZB tile 最远深度微变浅 → 剔除变难），M3.4 断言重校至 ≥1/6）
  - [x] M4.2 HW 光栅 pass + 路由 ✅ 2026-09-05（路由=UE SmallEnoughToDraw 简化版：簇最大边长(运行时从 index/position 池导出,computeClusterAabbs 扩为 3×vec4/簇)×实例最大缩放×(H/2)/(tanHalf×最近球面点距离) ≥ 18px(GNaniteMinPixelsPerEdgeHW 默认值)→ hwPairs(binding 10);params[7]=(hwMode,阈值),hwMode 0/1/2(2=FORCE 差分开关=UE RENDER_FLAG_FORCE_HW_RASTER);counters[7] 计数。硬件绘制=UE 同构:count_to_draw.slang(counters→VkDrawIndirectCommand{384,count})+非索引 Instanced Indirect Draw(VertexID/3=三角,VS 顶点拉取+越界三角形退化),**零附着管线**(createNullRaster:无颜色/深度目标,cull BACK/front CCW 与 SW 约定一致,fragmentStoresAndAtomics 特性已启用);PS 内嵌 rasterize_sw 的逐位覆盖判定(VS 经 flat varying 传 1/256 量化三角,PS 用 int64 边函数+DX top-left 重算——同时解决 helper-lane 污染与硬件 snap 语义依赖),深度位用 SV_Position.z(透视校正,与 SW 仿射的分歧 UE 同款已文档化)。验收:FORCE_HW vs all-SW 双链 4 帧覆盖集逐位一致(零洞零多余);密集场景 app 62 簇全路由 HW、occ 继续工作、校验层零错误、确定性复现;混合路由下 HZB 混合两种深度位→边界遮挡决策可漂移(保守方向,文档化)。one-shot 差分测试改 hwMode=0 纯 SW 对比）
  - [x] M4.3 async compute 重叠（P1 收尾）✅ 2026-09-05 深夜（commit 6accd38，v1：整链单命令缓冲上 compute-only 族 family 2 真异步引擎，自有 timeline；图形帧等链完成再 HW draw+decode → 跨帧流水重叠。实测 wall 场景 15/25/8 距离 3.0x/1.9x/3.4x 提速；47 用例绿、校验层零错误、画面逐字节一致、20k 帧长跑净。跨队列四件套：CONCURRENT 共享（雷点已记）、链间 timeline 排序、槽 pacing 扩展、readbackBuffer 补链等待；toFragment 屏障在 compute 族上剔除。落地挖出两 bug：HZB#2 warmup 帧的 vis 读取无序（旧 HW 段尾部屏障是唯一排序，重构移走后暴露——无条件 b(vis) 修复）、双链测试同帧同 cb 重 begin 竞态（每帧一链守卫，后续链回退图形队列）。v2 留档：同帧 HW||SW 并发需 cull/raster 拆段+双图形提交
  - [x] M4.4 场景验收 ✅ 2026-09-05 深夜（commit 序列末。大三角场景=kgeo_builder 加分段参数（--proc sphere:48,24 → out/lowpoly_sphere.kgeo，24k tris 全粗簇），近中距全 34 簇路由 HW（18px 阈值全超）✓；面板 HW 计数曲线入列 ✓；--no-async CLI + setAsyncChainEnabled API 做**同二进制干净 A/B** ✓。**async 诚实结论（修正 M4.3 提交里 3× 的跨二进制污染对比——HZB 归约等修复夹在测量中间）**：重链场景赢（knot 1080 vs 924=+17%、knot-dense 502 vs 474=+6%），轻链场景亏（sphere 系 -7~-14%，链占帧时零头时跨队列提交+信号量往返主导），wall@8 重载持平（445 vs 456）；async 开/关画面逐字节一致、校验层零错误、47 用例绿。**v1 的适用域=链占比大的场景**；轻场景默认关或自适应（后续优化项）
- [ ] M5 BVH8 + Persistent Threads 剔除
- [ ] M6 分页流式加载
- [ ] M7 量化压缩（八面体法线/隐式切线/拓扑编码）
- [ ] M8 Imposter / VSM / TAA
- [ ] Interfaces/ DLL 公共 API 边界层（参照 renderEngine，需要分发时）
- [ ] TextureManager（需要纹理时）

## Vulkan RHI v2 重构

（详见 docs/vulkan-rhi-plan.md，2026-09-01 调研定稿）

- [x] P0（M2 前置）：拆 CommandContext/DeletionQueue/DescriptorCache/UploadEngine + 每帧单一 rendering scope + staging ring ✅ 2026-09-02（工作块 0-3 落地、4(vk-bootstrap) 取消，详见 vulkan-rhi-plan.md §五）
- [x] P1（M4 前置）：timeline semaphore + compute 队列 + GPU timer ✅（timeline+GPU timer 随 P0 于 2026-09-02 落地；compute 队列随 M4.3 2026-09-05 落地）
- [ ] P2（M6 前置）：分页上传/传输队列/驻留预算

## 杂项待办

（随手记，格式：`- [ ] 事项 —— 备注/日期`）

- [x] HZB 遮挡体侧"盒光栅"调研 ✅ 2026-09-05 撤销：查证 nanite.md D4.1——**UE5 官方 HZB = 深度缓冲 min 归约金字塔（reversed-Z 下 2×2 取最远），无包围盒光栅进 mip 的操作**；边缘保守带（背景像素=0 污染 tile min）是官方接受的保守性，D4.3 明言"cluster 粒度 HiZ 无法像素级遮挡融合"是 Nanite 已知缺陷。我们的逐级归约 + ≤4×4 mip 测试已与 D4.1/D4.2 逐条一致，无算法差距。远距剔除收益的正解 = 场景侧大遮挡体（--wall 演示场景），不是改金字塔（曾误记"Nanite 式盒光栅"并当成既定技术路线，已纠正——教训：把想法标上引擎出处之前先查主文档核实）
- [x] **CONCURRENT 共享雷点复审 ✅ 2026-09-06**（用户 2026-09-05 判定雷点，本条为约定的对照 A/B）：KENSHIN_SAME_FAMILY_ASYNC 环境变量强制 async 链用同族第二队列（family 0 index 1，缓冲保持 EXCLUSIVE、零跨族共享）与默认跨族（family 2 compute-only，CONCURRENT）同二进制对照——**两种变体各自 47 用例全绿、画面逐字节一致**；性能：跨族 CONCURRENT 全面更快（knot 273 vs 250=+9%、wall@15 1153 vs 998=+16%、wall@25 1301 vs 1268=+3%）——同族第二队列共享图形引擎硬件、只有提交级并发且与 graphics 争引擎。结论：本机（RTX 5060 Ti）保留跨族 CONCURRENT，依据=自有测量而非"UE5 同款"（该出处说法维持撤回）；环境变量留作其他硬件复测入口。显式 ownership transfer 变体未做（同族对照已给出数量级：跨引擎收益 > 共享模式成本，transfer 的屏障税只会更差）

- [ ] CommandContext 记 last-bound 状态（pipeline/顶点/索引缓冲），`draw()` 跳过重复绑定——当前每帧 draw 数个位数、驱动对冗余绑定有优化，纯属顺手加固；改动只在 backend 内部，不动 API —— 2026-09-02
- [ ] **no-validation 呈现路径 DEVICE_LOST：重启不愈、疑似换驱动不愈，签名已收窄（2026-09-06 复测）**：2026-09-05 曾判"驱动劣化，重启后复测"，今日复测推翻劣化残留说——机器当日 16:39 全新启动、driverVersion 2559737856→2584739840（疑似换过驱动）、HEAD 2d1ec19 Release 构建：`--no-validation --frames 2000` 窗口呈现三次全灭（全链 / --no-occ / --no-async），死亡时刻高度一致（in-frame chain ready 后 ~6.0s；首个报错 = FrameSync.cpp:158 waitTimeline 的 vkWaitSemaphores——纯检测点非事发点，其后全部 submit 秒败 = 死设备空转模式）；**--no-occ 照死、--no-async 照死**——遮挡链与跨族 async 均排除，触发面收窄到"validation 关闭 + 呈现路径"（09-05 曾验证 S2 旧代码也死，跨代码版本）；**同参 validation 开 2000 帧零 DEVICE_LOST**。绕行 = 开发期保持 validation 开（M3/M4 全程如此运行）；候选下一步：DDU 干净重装驱动复测、Nsight/Aftermath 抓现场、或暂时搁置 —— 2026-09-06
- [x] one-shot 链入帧化 ✅（VisBufferChain 即此产物，M3 系列全量入帧；one-shot 路径留作差分参照——M2 后续不再需要，2026-09-06 清理）
- [ ] 软光栅差分盲区对策：GPU/CPU 差分两边共享实现缺陷时不可见（S4.4 半像素行偏移、双重 view 两个 bug 都是 S4.5 对 HW 基线对比才暴露的）；后续可加一条 float64 独立地面真值 raster 校验 —— 2026-09-02
- [x] M3.4 后退役 CPU 割切割渲染路径 ✅ 2026-09-06 全三步完成（依赖图 2026-09-04 盘点）：① ✅ 2026-09-05 —— kgeo 默认转正虚拟链（`--virtual` 保留解析为 no-op,历史验收命令仍可复现），基线路径降级 `--baseline` 开关（验证：无开关 kgeo 截图与显式 --virtual 逐字节一致、--baseline 路径可用、校验层零错误）；② ③ ✅ 2026-09-06（commit 801a5ec）—— pixelThreshold 滑条搬进 Virtual Geometry 面板（vgDebug，--kgeo 从 --lod-threshold 初始化；"切割三角形数"已随 M3.4 进链统计 readStats[11]）、updateCut/rebuildCutMesh/meshHandle 全删、KgeoPreview 缩为 dag 装载器+静态信息面板、--baseline 变 logged no-op（同 --virtual）；render_kgeo ctest 参考图自 M3.4 后过期未察觉，按文档流程重生成后两条 ctest 全过。**永留**：loadKgeo+dag（GPU 路径数据源头）、selectLodCut（LOD 语义地面真值——M5 换遍历结构时"新遍历集合==selectLodCut 集合"是最便宜的语义回归）
- [x] M3.2b 遮挡包围体收紧（提前，随链内遮挡接线一起做）✅ 2026-09-05（簇 AABB 8 顶点投影紧矩形替换球的 ±R 方形；AABB 运行时从簇顶点导出（每簇 2×vec4，.kgeo/GpuCluster 不动）；HZB 接入帧内链——每槽金字塔建于帧首、源是同槽滞后两帧的 VisBuffer（可证完备），首帧/resize 由 m_hzbValid 门控；occ-culled 计数入 readStats[8] 与调试面板；fillBuffer 自带 TRANSFER→COMPUTE 屏障。修：readStats/readIndirectArgs 首帧前无守卫致 app frame-0 崩溃、host 统计缓冲 + vis 缺 TRANSFER_SRC。验收：单测帧内遮挡保守精确（VisBuffer 逐位一致 + pair 守恒）；app 实心球沿轴机位 occ-culled 0→1；球/结两场景与无遮挡 HEAD 同参截图逐字节一致；校验层零错误（顺带消除 HEAD 的 binding-7 未更新遗留）。注：结类镂空几何 HZB mip 混洞深度、遮挡收益天然低，量化 RECT 收紧收益待 M3.3 后用大遮挡体场景测）
- [ ] （示例）调研 clusterlod.h 的 clodConfig 各参数含义 —— 对应 S3.1
- [ ] RenderView 抽象（多视口/多窗口时引入；Filament View=Scene+Camera、renderEngine RenderView 同位）—— 2026-09-01 Renderer/IRenderPipeline 落地时明确推迟
- [ ] RenderGraph（pass 图 + 资源生命周期 + 调度；参照 Frostbite FrameGraph / UE RDG / Godot RDG；UI 合成届时从 backend endFrame 上移为真 pass）—— M2 多 pass 化时引入
- [ ] Input 迁入子系统容器（当前 static 单例被 Window 的 WndProc 直接喂数据，解耦需动消息泵）—— 2026-09-01 子系统容器落地时评估
- [x] VulkanBackend：per-draw rendering scope 合并为每帧单一 scope（CommandContext::beginViewportPass/endPass，2026-09-01 P0-2 完成并字节级验证）
