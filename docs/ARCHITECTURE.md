# 目录与运行边界

- `run_real_vla.py`：命令行生命周期、模型服务、唯一身体 owner、原生终端、相机窗口和录像请求。
- `vla_camera_view.py`：独立相机订阅、仅本机且带随机令牌的预览服务、MP4 编码和录像审计。
- `runtime/plain_chunk.py`：普通 GR00T 动作分段队列，复制到每轮运行副本。
- `src/symbiosis/`：从原工程复制并收窄的真机运行代码。保留包名以保持协议及调用兼容，不依赖原仓库安装。`task_runtime` 仅提供本地任务名映射，没有 GPT、自动 planner 或仿真任务管理。
- `vendor/rtc/gr00t/`：本地实际使用的 GR00T 与 RTC 模型、处理器、客户端和服务代码。
- `vendor/sonic/`：沿实际 Python 导入链复制的 SONIC 模块、观测转换、相机协议、初始姿态、手部通信和手部配置。
- `vendor/unitree/`：Unitree Python SDK 副本。导入 SDK 不等于创建硬件通道；验证程序不执行初始化。
- `assets/controller/`：ONNX encoder、decoder、planner、当前 GPU 的 TensorRT engine 和观测配置。
- `assets/reference/`、`assets/robot_model/g1/`：原生参考动作及 URDF／网格。
- `native/bin/`：已固定哈希的 Linux x86_64 `g1_field` 及部署 manifest。
- `native/source/`、`native/thirdparty/`：对应原生源码和 SDK 头文件／静态组件。
- `native/lib/`：DDS 动态库及原生构建静态组件。CUDA、TensorRT、ONNX Runtime 等系统 SDK 不包含在仓库里。
- `configs/defaults.json`：仓库相对路径；`configs/local/g1.json`：忽略的机器配置；启动时冻结绝对路径到本轮目录。
- `provenance/`：导入文件来源及哈希、软件验收摘要。来源中的旧绝对路径用于追溯，不作为运行配置。

`runtime_config.py` 把仓库内的应用模块放在 Python 路径最前面。生成的 owner 使用本轮快照及仓库内 vendor 路径。`verify_standalone.py` 额外拦截关键应用包从旧 editable checkout 加载的回退。

目前可以继续使用机器上已有的 Python 环境提供第三方包；其解释器地址是本机配置，不是应用源码路径。换机器需安装等价依赖并填写本机配置。模型权重和固定 Cosmos 基础模型仍由用户提供。

原生硬件约束保留：DDS domain 0、eno1、CRC、350ms 传输租约、源时间戳、反馈新鲜度和唯一 owner。独立启动器在每轮副本中启用无限 VLA、动态文本、独立模型端口，以及操作员要求的 45° 倾角阈值。其他原生／设备保护仍可能提前停止。

本次整理只进行软件和离线验证，不把冻结配置、导入成功或 MP4 测试当作真机动作验收。
