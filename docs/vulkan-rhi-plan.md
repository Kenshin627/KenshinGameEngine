# Vulkan 后端重构计划（RHI v2）

> 2026-09-01 整理，同日补"API 封装层"章节。依据：现状审计 + 业界实践调研
> （Vulkan-Hpp / Granite / vkguide / NVRHI / Diligent / The-Forge / Filament /
> zeux / NVIDIA Do&Don't / Khronos Vulkan-Samples）。
> 目标：把当前"能跑但脆"的 Vulkan 封装升级为支撑 M2-M8 的可靠底座。
> 原则：服务里程碑、按需引入，不做投机性通用化（staged-plan 精神）。

## 一、现状审计（证据说话）

### API 封装层问题（本次重点）

| # | 问题 | 证据 | 后果 |
|---|------|------|------|
| A1 | **错误检查无纪律** | `vkCreatePipelineLayout`/`vkCreateImageView` 等多处返回值不查；查了的地方也只有 `!= VK_SUCCESS` 局部分支，无统一出口 | 失败被吞，后续调用在无效句柄上连环炸，定位回到"逐行猜" |
| A2 | **零调试可观测性** | 全仓无一处 `vkSetDebugUtilsObjectNameEXT`、无 `vkCmdBeginDebugUtilsLabelEXT` | RenderDoc 里全是匿名句柄和裸提交，抓帧时对不上业务对象 |
| A3 | **VkResult 无人类可读形态** | 日志/断言里只能打整型 | 出错时 "result = -4" 还要翻表 |
| A4 | **裸句柄 + 手工销毁顺序** | Impl 里 20+ 个 Vk 句柄成员，shutdown() 手工逐个 vkDestroy，顺序靠人脑记 | 漏销毁/顺序错就是泄漏或校验层警告；God-file 的主要成因 |
| A5 | **结构体构造已是较好状态** | C++20 指定初始化器 + 具名局部（好） | 保留——vulkan-hpp 反而不支持指定初始化 |

### 架构层问题（原有审计）

| # | 问题 | 证据 | 后果 |
|---|------|------|------|
| B1 | **God file** | `VulkanBackend.cpp` 1510 行：设备/上传/录制/回读/ImGui/管线全在一类 | 今天的深度/绕序 bug 都在此排查 |
| B2 | **per-draw rendering scope + 手写散落屏障** | 每个 `draw()` 自开 rendering scope；color 跨 scope 无屏障（todoList 挂账） | 规范未定义行为；M2 多 pass 必炸 |
| B3 | **即时销毁、无在飞保护** | `destroy*` 直接 vkDestroy | kgeo 调阈值即潜在 use-after-free |
| B4 | **描述符管理缺失** | 仅 ImGui 池，校验层整天刷 SAMPLED_IMAGE 警告 | M2 SSBO 大绑定不可用 |
| B5 | **上传路径原始** | 逐资源一次性 staging，无每帧动态数据通道 | M2 每帧剔除输入无通道；M6 无宿主 |
| B6 | **同步原语半吊子** | timeline semaphore 只探测未使用 | M4 async compute 零基础 |
| B7 | **无 GPU timer** | — | M3+ 调优无抓手 |

**骨架保留项**：FrameSync 2 帧在飞槽位、VMA、vulkan/ 子目录拆分、句柄式资源
（mesh/pipeline + 空槽复用）、sync2 屏障 API、指定初始化器风格。

## 二、Vulkan API 封装层设计（决策 + 规格）

### 2.1 路线决策：C API + volk + 自建薄封装（不换 vulkan-hpp）

| 维度 | vulkan-hpp / raii | C API + 薄封装（选定） |
|------|------------------|----------------------|
| 编译期 | 头文件巨大：无 PCH 时可占 TU ~90% 解析时间（[Khronos issue #1325](https://github.com/KhronosGroup/Vulkan-Hpp/issues/1325)、[SO 实测](https://stackoverflow.com/questions/71292498/)），需 PCH/modules 缓解 | volk 头很小，零负担 |
| 安全性 | 大量运行期错误变编译期错误（[NVIDIA 博客](https://developer.nvidia.com/blog/preferring-compile-time-errors-over-runtime-errors-with-vulkan-hpp/)） | 靠 VK_CHECK 纪律 + 校验层补 |
| 抽象漏损 | raii 句柄与手工生命周期（deletion queue、多帧在飞）配合别扭（[The Vulkan-HPP Dilemma](https://willrosecrans.wordpress.com/2021/05/31/the-vulkan-hpp-dilemma/)） | 与 DeletionQueue/帧槽模型天然契合 |
| 引擎界实践 | 工具/示例/独立游戏常见 | Granite / Filament / NVRHI 内部 / vkguide 一致选 C API |
| 教学价值 | — | KENSHIN 是学习工程，裸 API 可见性本身是目标 |

结论：**保持 volk + C API**，把"该包的"包进一个内部工具头，规模 ~200 行，
装得进脑内 RAM（[Tardif 原则](https://alextardif.com/RenderingAbstractionLayers.html)）。

### 2.2 `engine/src/backend/vulkan/Vk.h` 规格（新增）

```cpp
namespace kenshin::backend::vkutil {

// 1. 错误纪律：一切 VkResult 出口走这里
//    - debug 构建：失败即 KENSHIN_ERROR(file:line + 结果名) + assert 中止
//    - release 构建：同日志，返回 false 由调用方走失败路径
bool check(VkResult result, const char* call, const char* file, int line);
#define KE_VK(call) vkutil::check((call), #call, __FILE__, __LINE__)

// 2. 结果名：VkResult -> "VK_ERROR_OUT_OF_DEVICE_MEMORY"（A3）
const char* resultName(VkResult r);

// 3. 调试命名 + 命令标签（A2，RenderDoc/NSight 直接可读）：
//    - setName(device, VK_OBJECT_TYPE_BUFFER, u64Handle, "mesh-cube-verts")
//    - struct DebugLabel { ctor 记 begin / dtor 记 end }（cmd buffer 区段）
//    - 仅在 validation 开启时生效，零 release 开销
void setName(VkDevice, VkObjectType, u64, const char*);
struct ScopedDebugLabel { ... };

// 4. 句柄生命周期：不搞 UniqueHandle 包装（与多帧在飞模型冲突）。
//    生命周期归两条既有纪律管：设置期对象 = Impl 成员手工序（现有），
//    运行期资源 = DeletionQueue（架构节 B3 的解）。Vk.h 只提供
//    destroyIgnoring(VkResult)（销毁路径也可能返回错误，吞掉并记日志）。
//
// 5. 高频结构体默认值工厂（vkguide vkinit 式，收编非零工程默认）：
//    imageBarrier2() / imageView(...) / submitInfo(...) /
//    commandBufferBegin(...) / debugMessenger(...) / imageCreateInfo(...)
//    —— 只暴露 2-4 个真正的变量参数。
//    风格规则：一次性结构体（管线状态等）保持指定初始化器原样写，
//    镜像 spec、策略可见；工厂只服务高频复用处，禁止把一次性状态的
//    策略值藏进默认值（绕序 bug 的教训：状态要看得见）。
}
```

落点：现有全部 `vkXxx` 调用逐步套 `KE_VK()`（P0 第一步，机械替换 +
逐字节验证）；资源创建处补 `setName`（mesh/pipeline/depth/swapchain）；
`endFrame` 的 pass 区段包 `ScopedDebugLabel("opaque") / ("ui-composite")`。

### 2.3 设置期：保持自有 instance/device 创建（不引入 vk-bootstrap）

2026-09-02 用户裁定：**不引入 vk-bootstrap**，VulkanContext/VulkanDevice
的自有创建代码保持现状（已隔离在 vulkan/ 子目录、KE_VK 纪律覆盖、
caps 报告逻辑自有）。评估记录备查：vk-bootstrap（[仓库](https://github.com/charles-lunarg/vk-bootstrap)、
[vcpkg port](https://vcpkg.io/en/package/vk-bootstrap.html)）提供 builder 式
Instance/Device/Swapchain 创建与类型化 feature 链，vkguide 采用；当初
提议接管 InstanceBuilder/PhysicalDeviceSelector/DeviceBuilder、Swapchain
保持自有——该提议在计划评审时被否，理由：现有创建代码能跑且已模块化，
换库替换的收益（feature pNext 链工效）不抵迁移成本。

## 三、业界实践汇总 → 架构采纳矩阵

| 参考 | 关键机制 | 本项目采纳 |
|------|---------|-----------|
| [NVRHI](https://developer.nvidia.com/blog/writing-portable-rendering-code-with-nvrhi/) | 句柄化资源；自动屏障跟踪 | 句柄化（扩展到 buffer/texture）；自动屏障不抄（RenderGraph 的活） |
| [vkguide](https://www.vkguide.dev/docs/chapter-2/cleanup/)（[帧架构](https://www.vkguide.dev/docs/new_chapter_2/vulkan_new_rendering/)） | 每帧在飞一个 DeletionQueue；描述符缓存每帧 flush | 直接采纳 |
| [Granite](https://themaiter.net/blog/2019/04/23/a-tour-of-granites-vulkan-backend-part-4/) | 每帧线性 staging 分配器；C API + 断言纪律 | staging ring；API 纪律同源 |
| [Diligent](https://github.com/DiligentGraphics/DiligentEngine) / [The-Forge](https://github.com/confettifx/the-forge) | 现代 RHI 分层结构 | 结构参照，规模裁到单后端 |
| [Filament](https://github.com/google/filament) backend | Driver 命令流翻译 | 形态印证（已同构），多线程命令流不做 |
| [zeux](https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/)、[NVIDIA Do&Don'ts](https://developer.nvidia.com/blog/vulkan-dos-donts/) | 描述符分层；pass 合并录制 | 描述符分层；每帧单 scope |
| [Vulkan-Samples debug_utils](https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/extensions/debug_utils/README.adoc) | 对象命名/标签/messenger | 全套进 Vk.h |
| [Raptor 引擎](https://github.com/PacktPublishing/Mastering-Graphics-Programming-with-Vulkan/tree/main/source/chapter9/graphics)（Packt《Mastering Graphics Programming with Vulkan》ch9，SEED/Frostbite 血统） | 见下"Raptor 借鉴" | 印证 + 三项新增 |

**共识五条**：① 资源句柄化 + 延迟销毁；② 屏障集中管理（禁散落手写）；
③ 每帧在飞 = 自包含资源包（cmd + 删除队列 + staging + 描述符缓存）；
④ 上传走每帧线性分配器；⑤ Vulkan 1.4 特性该用就用（dynamic rendering
已用；push descriptor / timeline semaphore 待用）。

**Raptor 借鉴**（ch9/graphics，AAA 血统的最小实现，与本计划互证）：
- **印证既有计划**（API 形状可直接参考）：destroy 延迟/即时双轨
  （deletion queue + `*_instant`）；`dynamic_allocate(size)` 单环形缓冲 +
  帧偏移（= StagingRing）；timeline semaphore 跟踪 value 做 async compute
  提交（= P1）；`set_resource_name`/`push_marker`/GPU timestamps（= Vk.h + P1）
- **新增采纳 1：`access_*`/`query_*` 句柄三件套**——句柄对外的唯一面孔，
  内部经 `access_texture(h)` 取实体、`query_*(h, Description&)` 取元数据，
  禁止裸指针外泄；Resources 扩展时照此形状
- **新增采纳 2：shader 反射驱动描述符布局**（Raptor `spirv_parser`）——
  DescriptorSetLayout 从编译产物反射生成，消灭 shader 与 C++ 布局的手工
  双份维护（M2 SSOB 重绑定时的错配防线）。实现优先走 Slang 自带反射
  API，SPIR-V 解析为后备
- **新增采纳 3：管线磁盘缓存**（`create_pipeline(creation, cache_path)`）——
  VkPipelineCache + 文件落地，进 P1
- **记录不采纳**：帧图 JSON 数据驱动（我们的 RenderGraph 走 C++ 声明式、
  编译期类型化更合学习工程；但"资源生产者/消费者同名即边"的自动建边
  思路 M2 时采纳）；稀疏 page pool（M6 虚拟纹理时参考其 API 形状）；
  Creation 结构 fluent setter（一次性构造保持指定初始化器，见 2.2 风格规则）

## 四、目标架构

```
IRenderBackend（公共接口不动——renderer 层已解耦）
   │
VulkanBackend（瘦壳：生命周期 + 接口翻译，目标 <400 行）
   ├─ vulkan/Vk.h（新）───────── KE_VK / resultName / setName / DebugLabel
   ├─ vulkan/VulkanContext·VulkanDevice·Swapchain（保留，+FeatureChain）
   ├─ vulkan/FrameSync（扩展）── 帧槽 = cmd buffer + DeletionQueue
   │                              + StagingRing + DescriptorCache(flush)
   ├─ vulkan/CommandContext（新）── 每帧单次 rendering scope、pass 切换、
   │                              draw/compute/上传、集中屏障工具
   ├─ vulkan/UploadEngine（新）── 环形 staging + 一次性初始化队列
   ├─ vulkan/DescriptorCache（新）── 池 + 每帧缓存；布局分层（帧/材质/实例）
   └─ vulkan/Resources（扩展）── 句柄表：mesh + pipeline + buffer/texture
```

关键决策（同前）：每帧单一 rendering scope 与 RenderGraph 无缝衔接；
DeletionQueue 帧槽 fence 过后真销毁；StagingRing 16MB/槽起步、超限回退
一次性路径；**不做的**：多后端、多线程录制、NVRHI 级自动屏障。

## 五、分阶段落地（对齐 staged-plan）

### P0 —— M2 前置（5 个工作块）
0. **Vk.h 落地**：KE_VK 纪律全量替换 + setName/DebugLabel 接入 +
    高频结构体默认值工厂（纯机械，逐字节验证）——先立纪律再动结构，
    后续每块重构都在"错误必报"的网里做
1. VulkanBackend.cpp 拆分：CommandContext / DeletionQueue / DescriptorCache /
   UploadEngine 四个内部类（接口与渲染结果零变化）
2. 每帧单一 rendering scope + color/depth 屏障集中化（清掉跨 scope 无屏障欠账）
3. DeletionQueue 接管全部 destroy（kgeo 重建为验证场景，Raptor 延迟/即时
   双轨同款）+ StagingRing + `stagePerFrameData()`（Raptor `dynamic_allocate`
   同款）；**DescriptorCache 落地**：池分层 + ImGui SAMPLED_IMAGE 警告清零 +
   布局由 shader 反射生成（Slang 反射 API 优先）
4. ~~vk-bootstrap 接管 instance/device 创建~~（2026-09-02 取消——用户裁定
   不引入该库，设置期代码保持自有，见 2.3）
5. 验收：校验层**零警告**（含 SAMPLED_IMAGE 那条）、ctest 100%、
   基线/kgeo 截图逐字节不变

### P1 —— M4 前置
5. timeline semaphore 替换 binary 信号量组合；compute 队列 + 提交时间线
   （Raptor 形状：单 timeline + value 跟踪 + 独立 compute 提交）
6. GPU timestamp（每 pass 耗时进 DebugOverlay）+ **管线磁盘缓存**
   （VkPipelineCache 落地文件，Raptor `cache_path` 同款）

### P2 —— M6 前置
7. UploadEngine 扩展：分页上传、传输队列、驻留预算（MeshPool 对接）

## 六、风险与对策
- **拆分引入行为变化** → 逐字节截图标准 + 校验层全程开 + 每块拆完即验
- **KE_VK 替换波及面大** → 纯机械替换、分文件提交、不混入任何逻辑改动
- **单 scope 与 ImGui 合成顺序** → UI 过渡期保持 backend endFrame 现状，
  RenderGraph 到位一并上移
- **staging 尺寸** → 16MB/槽起步，保留一次性上传回退路径
