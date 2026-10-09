# 三物体并行触觉演示

球体、方块、不规则凸多面体同时在双手上按压、滑动和切换位置。随机种子为 42，腕部中心间距为 20 cm；轨迹覆盖左右手各 5 个指尖、5 个近节、5 个中节及 1 个掌面分区，共 32 个区域。三个物体共享这些目标，并非每种形状分别遍历 32 个区域。

物体只在开始时复位，随后通过虚拟弹簧施力，以 240 Hz PhysX 积分、60 Hz 采样。跨区域移动采用平滑抬升轨迹，视频结束前全部离开接触。

## 指尖六维力的统计范围

旧实现匹配整根手指的 link，再把合力与力矩转换到指尖坐标系；因此近节或中节受压时，指尖旁边的六维力面板也会有读数。现在接触传感器只匹配每个手指的 `DIP_rubber_link`，只汇总该软垫与目标物体之间的法向力和摩擦力。力矩原点和表达坐标系仍为软垫 link。

`tools/check_tip_wrench.py` 在 PhysX 中分别对左右食指近节和指尖进行按压。左右近节压阻峰值约为 0.369，此时对应指尖深度及六维力全部为零；压住指尖时，两手均出现压入深度和非零六维力。原始结果见 [tip_wrench_check.json](evidence/tip_wrench_check.json)。这项检查验证接触归属，不表示实机六维传感器标定。

## 多物体传感器

每个物体使用独立触觉后端。光学信号按指尖选择实际接触目标，同一个目标提供 RGB、深度和 Marker；不混合不同目标的 RGB 图像。当前演示限制同一时刻每个指尖最多与一个目标产生光学压入，违反该条件会立即报错。

压阻融合在扩散前对各物体的原始响应逐点取最大值，再执行一次扩散。输出仍为归一化模拟响应，不是经过实机辨识的压力、电阻或多点力叠加模型。PhysX 六维力则汇总所有目标在该指尖软垫上的真实接触力与力矩。

## 复现

```bash
python examples/parallel_demo.py --headless --device cuda:0 --fps 60 --seed 42 --no-data --output outputs/parallel
python tools/validate_parallel.py outputs/parallel
python tools/check_tip_wrench.py --headless --device cuda:0 --output outputs/tip_wrench_check.json
python -m pytest tests -q
```

`--no-data` 省略体积较大的完整 RGB/深度 HDF5，但保留逐帧 `telemetry.npz`，供覆盖、时间间隔、接触与释放检查。若需要完整传感器图像数组，去掉此选项。

## 本次验证结果

成片为 2560 × 1600、60 fps，共 1137 帧，时长 18.95 秒。32 个区域全部达到响应阈值，十个指尖均有 RGB 变化、Marker 位移、压入深度和六维力响应；三个物体均产生过指尖光学接触。同一指尖的同时光学接触目标数始终不超过 1。最后 12 帧的深度、压阻、Marker 与六维力均归零；逐帧数据有限、时间间隔为 1/60 秒，视频完整解码通过。八项单元测试通过。

证据：[覆盖与峰值](evidence/parallel_report.json)、[自动检查结果](evidence/parallel_validation.json)、[视频参数与 SHA-256](evidence/parallel_media.json)。抽查了接触、跨区域移动和释放画面，面板与物块接触部位对应，模型表面无触觉点叠加。
