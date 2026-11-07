# G1 GR00T 真机推理

独立的 G1 真机 VLA 运行仓库，包含模型推理代码、SONIC 观测与手部通信、身体 owner、Unitree Python SDK、原生身体控制程序及其源码、ONNX／TensorRT 控制资产、机器人模型、相机预览和自动录像。运行时不导入原 Symbiosis、Isaac-GR00T-rtc 或 WholeBodyControl checkout。

用户指定的 VLA 检查点、Cosmos 基础模型、Python 环境和 NVIDIA 系统 SDK 单独准备，不复制进 Git。机器人侧需要已有相机／DDS 服务，本仓库不替换机器人上的服务。

## 当前工作站启动

顶层启动器只用 Python 标准库；模型和控制子进程使用 `configs/local/g1.json` 中的 Python 解释器。此工作站的忽略配置已准备好，默认使用已安装的第三方包；应用源码全部从本仓库加载。

```bash
cd /home/kinetra-5090-01/Documents/Codex/2026-11-04/new-chat/outputs/gr00t-real-inference

python3 run_real_vla.py \
  /你的GR00T检查点目录 \
  --no-rtc \
  --instruction "Put the apple and banana from the wooden table into the basket, then carry the basket to the small black round table and set it down."
```

路径含空格时加双引号。`--instruction` 接收任意任务文本，应与权重能力一致。启动会打开专用第一视角相机窗口，加载指定权重并准备只读遥测；身体动作由操作员在终端触发。

| 按键（无需回车） | 功能 |
|---|---|
| `k` | 启动身体 owner 与原生控制；再次按下关闭身体控制。推理需先按 `p` 停止。 |
| `i` | 约 6 秒过渡到预设初始姿态并张开双手；执行中需等待完成。 |
| `p` | 开始持续 VLA 和录像；再次按下暂停并保存视频，身体控制保持。 |
| `t` | 输入新任务文本，回车提交，Esc 取消；运行中切换会暂停旧队列再从新观测继续。 |
| `q` | 暂停 VLA、保存录像、停止身体控制并关闭相机窗口与模型服务。 |

没有 `o` 操作员按键。关闭原生控制仍通过其私有终端接收 O 实现；q/k 不通过强杀原生进程代替确认退出。若原生停止未确认，保留 owner 并提示操作员处理。

## 推理模式

- `--no-rtc`：普通 GR00T，每段 40 步、50 Hz。执行完一段重新观测并推理，等待时保持末端姿态；不发送 RTC 前缀，不修改原 checkpoint 的配置。
- 不加 `--no-rtc`：配置缺少／为零的 `rtc_max_delay_steps` 使用普通模式；RTC 检查点使用 delay=8、request interval=30。当前支持 40 步预测、64 维 SONIC token 和左右各 6 维 BrainCo 手部。
- 连续运行没有步数或总时长预算。反馈过期、无效输出、关节限位等已有保护仍有效。用户指定的身体倾角阈值为 45°；写入本轮副本，不动态修改已启动进程。
- 给普通权重补 RTC／质量条件字段不能证明它接受过对应训练，本程序不自动修改用户检查点。

## 相机与录像

直接订阅 `192.168.123.164:5555` 的左眼 `ego_view`，通常为 640×360 RGB。预览、录像和 VLA 使用同一路图像，未额外去畸变。相机窗口为独立浏览器实例，q 只关闭它，不操作已有浏览器会话。

每次开始推理新建一段 MP4。停止、故障或 q 时封装并检查视频；失败保留 partial 文件和日志。按主机接收时间重采样为 30 fps，等待／断流间隙保持上一帧并记录重复与过期帧，不把等待时间加速跳过。

默认输出在：

```text
runtime/runs/standalone-vla-时间-编号/
  recordings/vla-时间-编号/
    camera.mp4
    recording.json
    frame-timing.jsonl
    vla-job.json
  model.log
  camera.log
  control-001/
```

终端会打印完整路径。输出、录像和本机配置不提交 Git。

## 软件检查（不启动机器人）

```bash
python3 run_real_vla.py /你的检查点目录 --no-rtc --check
python3 -m unittest discover -s tests -v
python3 scripts/verify_standalone.py --model /你的检查点目录
```

测试需要 NumPy、OpenCV、ffmpeg 和 ffprobe。相机测试使用明确标记的合成图像，不连接机器人。独立性检查阻止关键应用包从仓库外导入，读取机器人模型并生成临时控制副本，不初始化 DDS 通道、不开相机、不加载 GPU 权重、不运行控制器。

## 换机器和 Git

先安装 Git LFS，再拉取包含 LFS 对象的完整仓库：`git lfs pull`。本地资产已经复制并保存于此仓库，不是指向旧项目的符号链接。当前没有配置远程仓库或推送。

配置说明与系统依赖见 [docs/SETUP.md](docs/SETUP.md)，目录及依赖边界见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。来源和许可见 [NOTICE.md](NOTICE.md)、`licenses/` 与 `provenance/`。
