# 仓库维护约定

- 本仓库是真机 GR00T 执行工程。应用代码、控制资产和原生程序均位于仓库内，不恢复对 Symbiosis、Isaac-GR00T-rtc 或其他旧 checkout 的运行依赖。
- 先读 README.md、docs/ARCHITECTURE.md 和 provenance/VALIDATION.json。
- 权重、基础模型、Python 环境、凭据及现场录像不提交；configs/local/g1.json 为忽略的本机配置。控制器 ONNX、机器人模型和原生二进制需要提交，使用 Git LFS。
- 保留 k/i/p/t/q：k 开关身体，i 准备，p 推理开关，t 文本，q 保存并退出。没有操作员 o 按键；向底层发送 O 仍是 k/q 的明确停止实现。
- 不自动运行真实机器人或重启现场控制。优先 unittest、--check 和 scripts/verify_standalone.py；它们不发布机器人动作。
- 保留唯一 owner、请求身份/时效、实测反馈、关节限位、手部限位、原生 CRC/domain 0/eno1/350ms 租约及暂停行为。独立副本按用户设置使用 45° 倾角阈值。
- 代码调整使用模拟相机验证预览、录像封装和退出。不得把离线测试当成真机成功。
- vendor 和 native/source 是已复制的具体代码，修改时维护来源说明及校验记录。新代码不能悄悄回退到外部 editable 包。
