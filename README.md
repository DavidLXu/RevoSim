<h1 align="center">RevoSim: Scalable Multimodal Tactile Simulation for Dexterous Manipulation</h1>

<p align="center">
  <a href="https://davidlxu.github.io/RevoSim-web/">项目网站</a> ·
  <strong>中文</strong> ·
  <a href="README_EN.md">English</a>
</p>

RevoSim 面向 sim-to-real，将视觉外观、关节运动学、动力学与多模态触觉集成在同一双手模型中。环绕手指舞展示原始材质的光泽、色彩，以及动力学驱动的逐指波浪、快速手势切换和比耶侧摆；下方接触演示展示各触觉区域的响应。

<p align="center">
  <a href="docs/videos/revosim-orbit-60fps.mp4">
    <img src="docs/images/revosim-orbit.gif" width="640" alt="RevoSim: dynamic finger dance with a 360-degree studio orbit">
  </a>
  <br>
  <a href="docs/videos/revosim-orbit-60fps.mp4">观看手指舞视频（60 fps，15 秒）</a>
</p>

每手有 21 个转动关节、247 个压阻通道和 5 个指尖视触觉区域。接触演示中，双手掌心朝上，用户通过 Python API 或键盘控制物体，即可读取原始模拟 RGB、Marker 位移、深度、六维接触合力和压阻响应。

![RevoSim dashboard: optical panels beside the hands and hand-shaped pressure maps below](docs/images/dashboard.png)

[观看三物体并行触觉 Demo（60 fps，约 19 秒）](docs/videos/revosim-demo-60fps.mp4)

球体、方块和不规则凸多面体同时在双手上按压和滑动，随机切换位置与方向，在约 19 秒内覆盖十个指尖及全部近节、中节和掌面压阻分区。双手腕部中心间距为 20 cm。六维力只统计指尖软垫上的接触，按压其他指节不会被计入指尖读数。

## 环境

目标环境是 **Linux、Python 3.11、Isaac Sim 5.1.0、Isaac Lab v2.3.2、CUDA PyTorch 2.7.0**。需要 Isaac Sim 支持的 NVIDIA RTX GPU 和驱动。运行验证使用 Ubuntu 24.04 / RTX 5090 32 GB。

使用独立环境，避免修改已有的 Isaac Lab 环境：

```bash
conda create -n revosim python=3.11 -y
conda activate revosim
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
python -m pip install -e ../IsaacLab/source/isaaclab

# 在本仓库根目录执行；所有手模型和传感器文件已经在仓库内。
python -m pip install -e '.[test]'
python tools/doctor.py
```

RevoSim 自带精简 Kit 启动文件，只使用 Isaac Lab core；不需要安装 `isaaclab_tasks`、`isaaclab_assets`、RL 或 Mimic 扩展。固定版本避免默认分支切换到不同的 Isaac Sim/Python 版本。Isaac Lab v2.3.2 对应提交 `37ddf626871758333d6ed89cf64ad702aef127d0`。

基础环境安装也可参考 [NVIDIA Isaac Sim 5.1 文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) 与 [Isaac Lab v2.3.2 安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html)。已有兼容环境时只需安装本项目，不需要再次下载 Isaac Sim。

## 直接运行

```bash
# 最小数据读取示例：球体按压左手食指，打印真实传感器数组。
python examples/read_sensors.py --headless --device cuda:0

# 三物体同时按压与滑动；固定随机种子便于复现，覆盖全部触觉分区。
python examples/parallel_demo.py --headless --device cuda:0 --fps 60 --seed 42 --output outputs/parallel

# 逐个物体的长演示，包含跨掌抛跳。
python examples/slide_demo.py --headless --device cuda:0 --shape all --fps 30 --output outputs/demo

# 单独录制某种物体；支持 15 / 30 / 60 fps。
python examples/slide_demo.py --headless --device cuda:0 --shape sphere --fps 60 --output outputs/sphere

# 可视化交互：打开 Isaac Sim 和十指触觉面板。
python examples/interactive.py --device cuda:0 --shape sphere --output outputs/interactive

# 双手错拍波浪、快速手势切换、比耶侧摆和交替弹指；直接仿真并渲染 15 秒环绕视频。
python examples/finger_dance.py --headless --device cuda:0 --output outputs/finger-dance

# 导出 30 fps 循环 GIF；可选安装 gifsicle 进一步压缩文件体积。
python tools/make_orbit_gif.py outputs/finger-dance/revosim-finger-dance-60fps.mp4 outputs/finger-dance/revosim-finger-dance.gif
```

交互按键：`W/S` 前后、`A/D` 左右、`Q/E` 上下、空格释放物体、`R` 复位到左手食指上方并重新连接驱动、`Esc` 退出。键盘控制的是连接物体的虚拟弹簧目标，物体运动与接触由 PhysX 求解。

视频帧率表示仿真采样和播放速度；离线渲染不承诺达到实时 60 fps。

逐物体长演示自动完成接近、按压、往复滑动、释放以及跨掌抛跳。抛跳的起始速度表示外部发射冲量，飞行阶段只有重力和接触作用；它不是自主抓取策略，也没有把物体隐藏绑定到手指。除显式复位外，不逐帧传送物体。

长演示的 `--quick` 只检查左右食指及掌面，适合安装后快速调试，不作为十指覆盖验证。`--no-data` 省略完整 RGB/深度无损数据；并行演示仍保存用于检查覆盖和接触的 `telemetry.npz`。`--max-frames` 是开发截断选项，报告会明确标记为未完成。

## 面板与数据

主视图居中。左手的五指 RGB、Marker field、Depth map 和六维接触力/力矩面板在左，右手面板在右。底部两个压阻视图保留完整手形，在对应掌面和 MCP/PIP 区域点亮，区域框与标签说明每组传感器位置。

原手指传感器位置是近节/中节压阻与指尖视触觉，未虚构指尖压阻硬件。底部压阻图根据实际网格轮廓和采样点作正交投影，保持初始张开手姿态；每点始终对应同一原始通道。压阻配色在响应 1 处饱和，深度配色为 0–4 mm，Marker 箭头放大 4 倍；原始数据不裁剪或放大。可查看[指节压阻响应示例](docs/images/pressure.png)。

输出目录包括：

| 文件 | 内容 |
|---|---|
| `dashboard.mp4` | 同步的双手与十指传感器面板 |
| `observations.h5` | 原始浮点 RGB、深度、Marker、压阻、六维力、姿态与时间戳，gzip 无损压缩 |
| `calibration.npz` | Marker 原始顺序、有效性掩码、相机坐标与配准资源 |
| `sensor_layout.json` | 运行时压阻点位、法线、所属 link，以及关节名称和角度范围 |
| `metadata.json` | 单位、坐标系、左右手/手指顺序及数据范围 |
| `frames.jsonl` | 每一帧的时间与演示阶段 |
| `physics.json` | 运行时添加的碰撞几何及质量补全记录 |
| `report.json` | 覆盖范围、各通道峰值和释放状态 |
| `telemetry.npz` | 并行演示逐帧的力、压阻、Marker、深度峰值、接触标志、物体姿态与光学目标编号 |

数组的第一维按 `left, right` 排列，第二维手指按 `thumb, index, middle, ring, little` 排列：

| 字段 | 单帧形状 | 语义 |
|---|---|---|
| `rgb` | `[2,5,3,240,320]` | 原始模拟视触觉 RGB，float32，范围 0–1 |
| `depth` | `[2,5,240,320]` | 接触压入深度，米 |
| `marker` | `[2,5,100,3]` | 指尖软垫 link 局部坐标的位移，米；需结合有效性掩码 |
| `wrench` | `[2,5,6]` | 指尖软垫与目标物体的接触合力与力矩，`Fx,Fy,Fz,Tx,Ty,Tz`，N / N·m |
| `contact` | `[2,5]` | 是否存在该指尖软垫与目标的法向接触 |
| `pressure` | `[2,247]` | 扩散后的非负归一化响应，可大于 1；不是 Pa、N 或电阻值 |
| `pad_pose` | `[2,5,7]` | 指尖软垫世界位置与 wxyz 四元数 |
| `joint_pos` | `[2,21]` | 实际关节角，弧度 |
| `object_pose` | `[7]` | 第一个目标物体世界位置与 wxyz 四元数 |
| `object_poses` | `[物体数,7]` | 所有目标物体的世界位姿 |
| `optical_object_index` | `[2,5]` | 多物体模式下，各指尖光学信号对应的物体编号；无压入时为 -1 |

六维合力仅包含 `DIP_rubber_link` 软垫与目标物体之间的法向力和摩擦力，不汇总近节、中节或硬壳上的接触。力矩参考点是 `DIP_rubber_link` 原点，结果也表达在这个局部坐标系中；面板为便于阅读把 N·m 转为 N·mm，原始数据始终保存 N·m。

压阻通道顺序：中指 MCP 23 / PIP 4，食指 23 / 4，无名指 23 / 4，小指 23 / 4，拇指 18 / 4，掌面 117。可通过 `revosim.resources.pressure_regions()` 获取切片。

## Python API

先启动 Isaac App，再创建 `World`：

```python
import argparse
from isaaclab.app import AppLauncher
from revosim.launch import configure

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(configure(parser.parse_args()), multi_gpu=False).app
try:
    from revosim.world import World
    world = World(shape="sphere", device="cuda:0")
    point, normal = world.surface_point("left", "index", "tip")
    world.reset_object(point + normal * 0.025)
    world.drive_object(point + normal * 0.005)
    for _ in range(120):
        world.step()               # 默认 4 个 240 Hz 物理步
        data = world.observe()     # 所有返回数组都有独立所有权，可直接保存
        print(data["wrench"][0, 1])
finally:
    app.close()
```

`world.sensor_layout()` 返回传感器点位、坐标系和关节顺序，可直接 JSON 序列化。

`world.set_joints(side, {joint_name: radians})` 控制关节目标，`world.release_object()` 解除探针驱动。`World(shape=("sphere", "cube", "polyhedron"), hand_spacing=0.20)` 创建三物体场景；`drive_object(position, object_name="cube")` 分别控制目标。多物体模式对原始压阻响应逐点取最大值，再做一次扩散；每个指尖的 RGB、Marker 和深度来自同一个接触目标。当前光学后端要求同一时刻每个指尖最多接触一个目标，重叠时会报错，不合成虚假的多目标光学图像。

## 验证与建模范围

```bash
python -m pytest tests -q
python tools/check_tip_wrench.py --headless --device cuda:0 --output outputs/tip_wrench_check.json
python tools/validate_parallel.py outputs/parallel
python tools/validate_recording.py outputs/demo/sphere
python tools/validate_recording.py outputs/demo/cube
python tools/validate_recording.py outputs/demo/polyhedron
```

验证器检查完整阶段、双手十指所有模态、全部压阻分区、原始数组形状/有限性、时间间隔以及离开接触后归零。验证器还检查抛跳前段的加速度与目标掌面落点响应，并完整解码视频以核对帧数。逐物体演示的运行证据见 `docs/validation.md`；本次并行演示及指尖六维力隔离检查见 [并行演示验证](docs/parallel_validation.md)。

手模型原本是视觉资产，本项目在运行时添加碰撞和关节驱动：掌壳与指节硬壳使用 SDF，避免凸包填平凹陷；其他碰撞采用分解或凸包近似，软垫使用柔顺接触。保留源文件中已有的质量/惯性，未指定质量的固定附属 link 使用极小质量，防止引擎默认质量改变动力学。驱动参数用于演示，并非实机辨识结果；自碰撞关闭，手腕固定。

压阻通道数量、顺序和切向位置保留。原资源的手指法线朝内，因此运行时翻转；埋在皮肤内的点沿法线投射到高保真软垫外表面作为采样位置。RGB 来自 Taxim，Marker 来自 HydroShear，视触觉配准是模拟资源，不代表实机标定；可视表皮仍是刚性网格，不是有限元软体变形。

## 项目结构

```text
src/revosim/       World API、场景、面板、记录器、精简启动文件
  tactile/        抽取的压阻 / 深度 / HydroShear / Taxim 后端
  assets/         左右手 USD、PBR 贴图、触觉资源与来源哈希
examples/         数据读取、三物体演示、键盘交互
tools/           环境检查、数据验证和诊断
tests/           通道契约、坐标配准、六维合力及独立性测试
```

第三方许可和文件声明详见 `THIRD_PARTY_NOTICES.md` 及各资源目录。

## 胶面参考与内部黑点修复

`TactileSensorCfg.outer_surface_reference` 默认是 `True`。深度通道沿射线选择最外侧朝外的胶面交点，替代固定取第二个交点，避免内部薄结构导致参考距离错误。胶面参考在初始化时计算并缓存；运行时物体仍使用最近交点，不会每帧重新遍历胶面。

`marker=True` 可以保留：marker 通道仍使用原标定和原交点规则，这个开关只改变深度胶面参考。设为 `outer_surface_reference=False` 可对比旧方法。此移植没有修改局部窗口插值、极端擦边交点步进或双环境差异问题。

回归命令：`PYTHONPATH=src python -m pytest tests -q`。单元测试验证交点选择和配置路由；完整动态仿真仍需兼容的 Isaac Sim 环境。

## 引用

如需引用本项目，可使用以下 BibTeX：

```bibtex
@misc{zhang2026revosim,
  title = {{RevoSim}: Scalable Multimodal Tactile Simulation for Dexterous Manipulation},
  author = {Zhang, Ke and Xu, Lixin and Wang, Ziyi and Dong, Xiyue and Tan, Jie and Li, Chuanyu and Xu, Renjing},
  year = {2026},
  url = {https://davidlxu.github.io/RevoSim-web/}
}
```
