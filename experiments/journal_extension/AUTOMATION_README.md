# 实验自动化流程说明

`experiment_flow_controller.py` 负责监督注册实验队列。它每分钟检查正式产物、JSON 可解析性和断点文件，并更新：

- `artifacts/journal_extension_20260814/automated_flow_state_v1.json`
- `artifacts/journal_extension_20260814/AUTOMATED_EXPERIMENT_FLOW.md`
- `artifacts/journal_extension_20260814/stage_reports/<阶段>.md`

主训练仍由 `run_remaining_pipeline.py` 按冻结顺序串行执行。控制器检测到已有训练或评估进程时不会启动第二个 GPU 作业；主流水线退出且存在未完成阶段时，会从第一个缺口调用同一入口，并沿用 `last.pt` 断点。所有输出和错误日志写入 `artifacts/journal_extension_20260814/logs/`。

启动常驻控制器：

```powershell
.\.venv-partc-win\Scripts\python.exe -m experiments.journal_extension.experiment_flow_controller --interval 60 --device cuda:0 --workers 12
```

当前实验只使用本地 TrajAir、TartanAviation-KAGC 和 TartanAviation-KBTP 数据。Tartan 测试是内部回顾性锁定评估；TrajAir aWTA 扩展只运行开发集，不构成第三方盲测或前瞻性确认。首次 aWTA KAGC 测试曾因空的可选 `interactive` 分组失败，已在不改变主指标的前提下修复并记录于运行日志，不能把该测试表述为完全未经触碰的盲测。
