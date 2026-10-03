"""Generate evidence-bounded C127 terminal artifacts and audit documents."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/metric_exact"
AUDIT_ROOT = ROOT.parents[1] / "审核文件"
REPORT = AUDIT_ROOT / "ASCENT_C127_最终报告_20260804.md"
PAPER = AUDIT_ROOT / "ASCENT_C127_论文框架_20260804.md"
LIVE = AUDIT_ROOT / "ASCENT_C127_实时进度_20260804.md"
METHOD_AUDIT = AUDIT_ROOT / "C1-C123_方法价值审计_20260804.md"
CLAIM_GUARD = ROOT / "experiments/metric_exact/locked_claim_guard_addendum.json"
START = "<!-- C127_AUTO_RESULTS_START -->"
END = "<!-- C127_AUTO_RESULTS_END -->"


def read_optional(name: str) -> dict[str, object] | None:
    path = ARTIFACT_ROOT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def decision(
    p1: dict[str, object] | None,
    p2: dict[str, object] | None,
    p3: dict[str, object] | None,
    locked: dict[str, object] | None,
) -> tuple[str, str]:
    if locked is not None:
        passed = bool(locked["locked_confirmation_passed"])
        return (
            "C127_CONFIRMED" if passed else "C127_LOCKED_NOT_CONFIRMED",
            "locked_test",
        )
    if p3 is not None:
        return str(p3["decision"]), "P3"
    if p2 is not None:
        return str(p2["decision"]), "P2"
    if p1 is not None:
        return str(p1["decision"]), "P1"
    return "C127_INCOMPLETE", "P0"


def p1_table(p1: dict[str, object] | None) -> list[str]:
    if p1 is None:
        return ["P1 尚无完整正式结果。"]
    lines = [
        "| arm | minADE | minFDE | p95 FDE | effective modes |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, metrics in p1["metrics"].items():
        lines.append(
            f"| {variant} | {float(metrics['minade']):.6f} | "
            f"{float(metrics['minfde']):.6f} | {float(metrics['p95_fde']):.6f} | "
            f"{float(metrics['effective_modes']):.3f} |"
        )
    return lines


def result_section(summary: dict[str, object]) -> str:
    p1 = summary["P1"]
    p2 = summary["P2"]
    p3 = summary["P3"]
    locked = summary["locked_test"]
    lines = [
        START,
        "## C127 实际执行结果",
        "",
        f"自动更新时间：{summary['generated_at']}  ",
        f"当前终端决策：`{summary['decision']}`（最远阶段：{summary['furthest_stage']}）。",
        "",
        *p1_table(p1),
        "",
    ]
    if p1 is not None:
        lines.append(
            f"P1 选择：`{p1.get('P2_selected_exact_candidate')}`；"
            f"决策：`{p1['decision']}`。"
        )
    if p2 is not None:
        lines.extend(
            [
                "",
                f"P2 同向 fold 数：{p2['same_direction_folds_vs_B2']}/5；"
                f"决策：`{p2['decision']}`。",
            ]
        )
        diagnostic = p2.get("untouched_replication_diagnostic")
        if diagnostic is not None:
            minade_ci = diagnostic["paired_date_bootstrap_vs_B2"]["minade"][
                "ci95"
            ]
            minfde_ci = diagnostic["paired_date_bootstrap_vs_B2"]["minfde"][
                "ci95"
            ]
            lines.append(
                "P2 后置非门控独立性诊断（仅 folds1-4）："
                f"同向 fold={diagnostic['same_direction_folds_vs_B2']}/4，"
                f"date={diagnostic['date_count']}；ADE/FDE paired-date 95% CI="
                f"[{minade_ci[0]:.6f}, {minade_ci[1]:.6f}]/"
                f"[{minfde_ci[0]:.6f}, {minfde_ci[1]:.6f}]。"
            )
    if p3 is not None:
        candidate = p3["candidate"]
        candidate_metrics = p3["aggregates"][candidate]
        lines.extend(
            [
                "",
                f"P3 候选 `{candidate}` 五 seed 均值：minADE="
                f"{candidate_metrics['minade']['mean']:.6f}，minFDE="
                f"{candidate_metrics['minfde']['mean']:.6f}；"
                f"同向 seed={p3['same_direction_seeds_vs_B2']}/5；"
                f"决策：`{p3['decision']}`。",
            ]
        )
    if locked is not None:
        candidate = locked["candidate"]
        candidate_metrics = locked["aggregates"][candidate]
        lines.extend(
            [
                "",
                f"一次 locked-test event：候选五 seed 均值 minADE="
                f"{candidate_metrics['minade']['mean']:.6f}，minFDE="
                f"{candidate_metrics['minfde']['mean']:.6f}；"
                f"确认通过={locked['locked_confirmation_passed']}。",
            ]
        )
    lines.extend(
        [
            "",
            "声明边界：P1/P2 是 train-only 开发，P3 是重复使用 dev 的开发复制；"
            "只有 `C127_CONFIRMED` 才表示一次 sealed locked-test 确认。",
            "P2 独立性边界：冻结的 5-fold 门与 74-date CI 包含用于候选选择的 fold0；"
            "只有 folds1-4 未参与候选选择，其单独统计是看到部分 P2 结果后增加的"
            "后置非门控诊断，不得表述为预注册门。",
            "确认边界：`C127_CONFIRMED` 同时要求原始 B2 三门与后置 B0 三门通过；"
            "B0 claim guard 在已查看 P1/部分 P2 后增加，只能表述为保守的后置加严，"
            "不能表述为原始预注册。",
            "次要指标限制：P1/P2 的 `tail_minfde` 阈值由全 train targets 计算，"
            "包含当前验证折，因此仅作探索性描述；该阈值不进入模型训练、候选选择、"
            "ADE/FDE 主门或 bootstrap。",
            "主张范围：即使确认，也只支持固定 K=5 的 oracle 几何精度；"
            "不自动支持 top1 排序、概率校准、分布质量、跨数据集泛化或评分梯度冲突的因果结论。",
            END,
        ]
    )
    return "\n".join(lines)


def replace_section(path: Path, section: str) -> None:
    content = path.read_text(encoding="utf-8") if path.is_file() else ""
    if START in content and END in content:
        prefix = content.split(START, 1)[0].rstrip()
        suffix = content.split(END, 1)[1].lstrip()
        updated = f"{prefix}\n\n{section}"
        if suffix:
            updated += f"\n\n{suffix}"
    else:
        updated = content.rstrip() + "\n\n" + section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def update_method_audit(summary: dict[str, object]) -> None:
    if not METHOD_AUDIT.is_file():
        return
    content = METHOD_AUDIT.read_text(encoding="utf-8")
    content = content.replace(
        "# ASCENT C1-C126-R1 方法与价值审计",
        "# ASCENT C1-C127 方法与价值审计",
    )
    row = (
        f"| C127 | metric-exact score-isolated ASCENT | "
        f"终端决策 `{summary['decision']}`，最远阶段 {summary['furthest_stage']}；"
        "直接对齐独立 minADE/minFDE，并以 B2 评分隔离复现作为强对照 | "
        "区分损失-指标错位、评分梯度与正速度坐标的贡献；无论正负结果均保留日期/seed 复制证据 | "
        "V1 `artifacts/experiments/metric_exact/final_summary.json`, "
        "`审核文件/ASCENT_C127_最终报告_20260804.md` |"
    )
    lines = [line for line in content.splitlines() if not line.startswith("| C127 |")]
    insert_at = next(
        (index for index, line in enumerate(lines) if line == "## 总结性判断"),
        len(lines),
    )
    while insert_at > 0 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, row)
    METHOD_AUDIT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run() -> dict[str, object]:
    protocol = load_protocol()
    p1 = read_optional("p1_summary.json")
    p2 = read_optional("p2_summary.json")
    p3 = read_optional("p3_summary.json")
    locked = read_optional("locked_test_result.json")
    claim_guard = json.loads(CLAIM_GUARD.read_text(encoding="utf-8"))
    terminal_decision, furthest = decision(p1, p2, p3, locked)
    summary = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_sha256": sha256(protocol.path),
        "decision": terminal_decision,
        "furthest_stage": furthest,
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "locked_test": locked,
        "locked_claim_guard_addendum": claim_guard,
        "locked_test_used": locked is not None,
    }
    output = ARTIFACT_ROOT / "final_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    section = result_section(summary)
    report = "\n".join(
        [
            "# ASCENT C127 最终报告",
            "",
            section,
            "",
            "## 证据索引",
            "",
            f"- protocol SHA256：`{summary['protocol_sha256']}`",
            f"- original locked analysis plan：`experiments/metric_exact/locked_analysis_plan.json`",
            f"- conservative B0 claim guard：`experiments/metric_exact/locked_claim_guard_addendum.json`",
            "- P0：`artifacts/experiments/metric_exact/preflight.json`、"
            "`deterministic_smoke.json`、`provenance.json`",
            "- 正式阶段代码修订：`artifacts/experiments/metric_exact/"
            "provenance_amendment_20260804.json`",
            "- 阶段汇总：`p1_summary.json`、`p2_summary.json`、"
            "`p3_summary.json`（仅在获授权时存在）",
            "- locked test：`locked_test_result.json` 和 receipt（仅在 P3 全门通过时存在）",
            "- 针对性文献复核：`审核文件/文献检索/C127_20260805/检索说明.md`；"
            "OpenAlex 检索有明确记录上限，不是系统综述",
        ]
    )
    REPORT.write_text(report.rstrip() + "\n", encoding="utf-8")
    replace_section(PAPER, section)
    replace_section(LIVE, section)
    update_method_audit(summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
