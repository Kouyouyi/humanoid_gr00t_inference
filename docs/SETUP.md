# 环境与部署

本仓库包含所需应用代码、控制资产、原生二进制和 SDK 源码，不需要克隆或安装其他机器人项目。仍需正常的操作系统、Python 第三方包和 NVIDIA／DDS 运行库。

## 当前工作站

已创建忽略文件 `configs/local/g1.json`，复用本机已安装模型、控制、遥测环境中的第三方依赖。所有应用代码由本仓库覆盖提供。VLA 权重由启动参数指定；Cosmos 基础模型固定到本机原有目录，不复制其参数。

顶层启动器只用标准库，直接 `python3 run_real_vla.py ...` 即可，不必先切换顶层环境。要手动切换解释器或基础模型路径，编辑上述忽略配置。不要编辑 provenance 中的历史路径。

## 新机器

1. 使用 Linux x86_64、支持 Blackwell 的 NVIDIA 驱动／CUDA 12.8，以及匹配的 TensorRT 10、原生 ONNX Runtime 1.16.3。随仓库提供的原生二进制是本机架构版本；其他 ABI／架构需重新构建并验证。
2. 安装 Python 3.12、ffmpeg、ffprobe、Chrome／Chromium 或 Firefox；桌面终端需有 DISPLAY 或 WAYLAND_DISPLAY。原生包装器使用 flock 和 util-linux 的 script。
3. 准备独立的模型环境与控制环境。模型依赖参考 `requirements/model.txt`；控制依赖参考 `requirements/control.txt`。这些来自当前运行环境与原模型依赖，不表示已验证全新机器的一键安装。Flash Attention 需与 PyTorch/CUDA/Python ABI 对应。
4. 复制 `configs/local.example.json` 到 `configs/local/g1.json`，填写 Python 解释器和 Cosmos 基础模型目录。权重参数不纳入仓库。
5. 准备与部署一致的 Unitree／CycloneDDS 基础库；Python SDK 源码已在 vendor。机器人通过 eno1 提供 LowState、secondary IMU、BrainCo 双手 DDS 以及 5555 相机流。仓库不自动配置网卡、不安装机器人端服务。
6. 首先运行 README 的 --check、unittest、verify_standalone.py，再按现场流程验证。不要把软件检查当成站立或抓取验证。

原生运行库路径在 `native/env.sh`，支持 `TensorRT_ROOT`、`onnxruntime_ROOT`、`CUDAToolkit_ROOT` 环境变量覆盖，只包含 SDK 路径，不 source 其他应用的 env 脚本。

## 原生重建

源码、头文件、固定构建清单和必要静态组件已复制。以下只编译文件，不连接机器人，不覆盖当前 native/bin：

```bash
source native/env.sh
python3 scripts/rebuild_native.py --dry-run
python3 scripts/rebuild_native.py --output native/build
```

新二进制和 manifest 在 `native/build/`。需要现场切换时，在已停止机器人控制的条件下设置本机 `external_paths.native_build`，重新执行软件检查及硬件验证。SDK 主版本、平台或基础静态组件发生变化时，需要相应重建／适配，不能承诺旧二进制跨机器直接运行。

## 大文件

控制器 planner 约 738 MiB，连同 encoder/decoder、机器人网格和原生文件使用 Git LFS。首次推送时必须上传 LFS 对象，克隆后执行 `git lfs pull`。VLA checkpoint、Cosmos 权重、环境、凭据和录像被忽略。
