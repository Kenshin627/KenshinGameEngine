# 引擎着色器（Slang 源）

S1.4 起存放 `.slang` 源，由 ShaderManager 运行时编译到 SPIR-V。

- `test_atomic.slang`（S1.4）——uint64 原子 + reduce 冒烟
- `baseline.slang`（S2.2）——基线前向渲染 VS/FS
- `cull_instances.slang` / `cull_clusters.slang` / `rasterize_sw.slang` / `decode_material.slang`（Phase 4）
