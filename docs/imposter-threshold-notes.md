# M9.2 Imposter 阈值路由 — 未定位的 DEVICE_LOST（调查笔记）

状态：原型已写完并验证过视觉效果，但 vgchain 测试套件 DEVICE_LOST，
已回退到静态停泊版（c869e25/35a240b，全绿）。本文件是下次续做时的现场记录。

## 原型的内容（工作树曾包含，revert 前）

1. `cull_instances.slang`：binding 4 = tinyParams（复用 planes 缓冲的
   [6]=(camPos, thresholdPx)、[7].x=1/worldPerPixel）。`castShadows==2`
   标记 imposterable，投影半径 < 阈值则实例跳过簇管线。
2. `VisBufferChain`：planes 缓冲 6→8 vec4；ChainInputs 加
   `imposterThresholdPx`（<=0 关闭）；dispatch 4→5 句柄（第 5 个
   再绑 planes）。
3. `InstanceCullPass` 一次性路径：栈上 planes8[8]（[6].w=0 关闭），
   job.buffers[4] 同一指针。
4. app：makeInstFor casts=2 标记带图集的画廊实例；CPU 侧四边形
   构建镜像同一公式（r/d/worldPerPixel < thresholdPx）。

## 验证过的部分

- **视觉效果成立**：干净 no-TAA A/B = 4495px 恰好落在立方体屏幕区
  （与静态停泊路由相同的数字）——GPU 剔除跳过 + CPU 四边形补位的
  协同是工作的。
- **默认路径位级不变**：阈值 0 时新二进制 vs 旧二进制 0px。

## 失败现场

- `[vgchain]` 9/9 全挂：chain init（"persistent queue: 64 entries"）
  后 ~5 秒 `vkWaitSemaphores → VK_ERROR_DEVICE_LOST`（FrameSync.cpp:124
  / BufferStore.cpp:207）——**校验层零报错的硬崩类**（M4.2/M6.3 的
  "多级 SSBO 寻址无钳制"家族）。
- P0 / hzb dense-stack 也挂（同样走链）。
- **app 不崩**：同二进制的无头截图跑通（130 帧 × 多次）。app 与测试的
  差异未定位。

## 下次的排查线索（按嫌疑排序）

1. **绑定 4 的描述符集布局**：compute 管线反射出 5 个绑定；测试里
   管线缓存（kenshin_pipeline.cache）可能残留 4 绑定版——先删缓存重跑。
2. **一次性路径的栈上 planes8**：Job 输入指针的生命周期假设——把
   planes8 改成 static/堆存对照一次。
3. **5 句柄数组**：确认 dispatchCompute 对 handle 数 > 反射绑定数时
   的行为（多传是否安全——clamp 还是越界写描述符）。
4. 排除法：把 cull_instances 的 tiny-skip 分支体注释（保留 binding 4
   声明与 5 句柄）重跑——区分"绑定存在"与"分支执行"两段。

## 已知的独立小尾巴

阈值构建 vs 参考图差 11270px（默认路径）——但新 vs 旧二进制 0px，
说明参考图本身可能滞后于某次预期变化（last regen 在 4fd4c26）；
下次重生成参考前先用 no-hw 对照确认。

## 静态停泊版（当前 HEAD）与阈值版的取舍

静态版把 debug 旗标映射为"park 全部 imposterable 实例"（一次性链
init），测试全绿、视觉证明同 4495px。阈值版是正确的生产形态
（按距离动态路由），但需要上面的 DEVICE_LOST 破案后再上。
