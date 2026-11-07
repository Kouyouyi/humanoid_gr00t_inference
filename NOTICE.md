# 来源与第三方声明

此仓库整理自操作者当前使用的真机执行链。应用和资产已复制，不通过 Git submodule、软链接或旧 checkout 路径提供运行代码。

- `vendor/rtc/gr00t`：Isaac-GR00T 的当前本机版本，包括本地 RTC 适配。保留各文件版权头，Apache-2.0 许可文本在 `licenses/Isaac-GR00T-LICENSE`。
- `vendor/sonic`、部分原生组件：GR00T-WholeBodyControl 的实际执行路径；上游许可在 `licenses/GR00T-WholeBodyControl-LICENSE`。
- `vendor/unitree`：Unitree SDK2 Python 副本；许可在 `licenses/Unitree-Python-LICENSE`。原生 Unitree／其他第三方组件的许可文件随 `native/thirdparty` 保留。
- `src/symbiosis`：用户本地 Symbiosis 真机运行代码的收窄副本，未包含 planner/GPT、仿真、经验数据库、凭据或运行结果。
- `native/bin`、`assets`：用户工作站现有的固定部署程序和控制资产。原始来源、文件大小与 SHA256 记录于 `provenance/imported-files.json`。
- `native/lib` 的 CycloneDDS 动态组件来自本机已安装 DDS 运行库。其源代码／许可参考 `native/thirdparty/unitree_sdk2/thirdparty` 随附组件；实际采用版本记录在独立性检查输出。

本仓库不替第三方代码或用户资产授予额外权利，也未设置公共远程仓库。准备向外分发时，需按各组件原许可及资产使用权限处理。VLA／Cosmos 权重没有复制或纳入 Git。
