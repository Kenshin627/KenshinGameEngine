# M9.2 Imposter Stage B — 运行时注入（实施方案）

状态：设计定稿（2026-09-30）。Stage A/A2 已落地：CPU 烘焙（engine/src/geometry/ImposterBaker.cpp）
+ .kgeo v9 序列化（--imposter）。本文件是注入路径的施工图。

## 已定型的关键决策（排障中验证）

1. **单簇切割限制（v1）**：imposter 只适用于最深层级恰好是**单个簇**（≤128 三角）的网格。
   多簇切割的三角序号跨簇，载荷的 7 位 tri 无法直接寻址。bake 侧已按 deepest-level
   簇集烘焙；Stage B 在加载时检查 `cut 大小 == 1`，否则视为无图集（空平面回退）。
   运行时表项直接记 `(instance, 全局簇索引)`——**解码路径零改动所需的一切都在
   现有 cluster 表里**（paged 寻址、材质 id、UV 全部免费工作）。

2. **载荷路由**：`listSel=3`（现有 0/1/2 之外的空闲值）→ 解码读 `imposterPairs[pairIdx]`
   （新增一个小 SSBO，CPU 每帧填充 `(instanceIdx, rootClusterIdx)`）。
   decode_material.slang 的 pair 解析处加一行分支；fsShadowDepth 不改（阴影链照旧走簇）。

3. **注入 pass 形态**：主链 raster 之后、HZB/decode 屏障之前，一个 compute dispatch。
   每个微小实例一个 workgroup 盖其屏幕矩形；每像素：
   - 视线方向 → 八面体映射 → 图集 nearest 采样（图集作 **SSBO** u32 平面，不做纹理——
     tri 不能双线性，SSBO 免新建纹理基建）
   - 命中：由 tile 坐标 + 存储 z 重建世界位置（`center + right·tx·he.x + up·ty·he.y +
     fwd·(z·2-1)·he.z`，包围盒半边随表项传）→ 主相机 viewProj → NDC z → 反位深度位
   - `InterlockedMax(visBuffer[i], (pairIdx<<9)|(3<<7)|tri)`——与主链同一 VisBuffer 原子
   - miss（0xFFFFFFFF）不写

4. **微小实例路由**：实例剔除 pass 里投影包围球 < 阈值（~12px）的实例**不进** cluster
   链（emitCluster 早退），CPU 侧同步把它们记入 imposter 表。阈值走面板滑条。
   注入 pass 需要的 per-instance 数据：屏幕矩形、包围盒 center/he、atlas 绑定
   （meshId → 图集 SSBO 区间）。

5. **已知缺口（v1 接受并记录）**：imposter 簇的页不被 page_touch 触碰（imposterPairs
   不在 touch 遍历里）——根簇页可能被 LRU 驱逐。缓解：v1 在 app 侧把根簇 pair 一次性
   塞进主 pairs（每实例一个 pair 的代价可忽略，touch 覆盖），或 StreamingManager 加
   "pin 根簇页"。与既有 post-pair touch 缺口同类。

## 施工顺序

1. `imposterPairs` SSBO + decode listSel=3 分支（最小改动，单测：vgchain 差分）
2. 注入 pass（新 compute 管线 + 图集 SSBO 上传 + app 帧内接线）
3. 实例剔除路由 + 阈值面板
4. 验收：多实例远场场景（grid 拉远），imposter 实例与簇渲染像素级对照 +
   切换距离无可见 pop（阈值滞回）
5. 阴影链不动（VSM 走簇，imposter 实例的阴影仍是簇级——UE 同款取舍）
