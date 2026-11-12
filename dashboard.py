"""Read-only local monitoring UI; reads public artifacts only."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FedCrop Monitor", page_icon="🌱", layout="wide")
st.title("🌱 FedCrop")
st.caption("联邦作物病害分类 · 训练进度、隐私预算与设备部署")
root = Path("runs")
available = sorted(p.parent for p in root.rglob("config.json")) if root.exists() else []
if not available:
    st.info("尚无运行记录。先运行 README 中的训练命令。")
    st.stop()
selected = st.sidebar.selectbox("实验", available, format_func=lambda p: str(p))


def read(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


@st.fragment(run_every="3s")
def live():
    history = read(selected / "rounds.json", [])
    summary = read(selected / "summary.json", {})
    if history:
        last = history[-1]
        columns = st.columns(4)
        columns[0].metric("完成轮次", last["round"])
        columns[1].metric("验证 Macro-F1", f"{last['macro_f1']:.3f}")
        columns[2].metric("模拟累计时间", f"{last['simulated_seconds']:.1f} s")
        columns[3].metric("中止比例", f"{last['abort_rate']:.0%}")
        st.line_chart(pd.DataFrame(history).set_index("round")[["macro_f1", "coverage"]])
        with st.expander("轮次与故障记录"):
            st.dataframe(pd.DataFrame(history), hide_index=True)
    else:
        st.info("等待首轮结果，或当前为集中式／本地训练对照。")
    events = []
    try:
        for line in (selected / "events.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A concurrent writer may not have completed the last line yet.
    except OSError:
        pass
    privacy = [e for e in events if e.get("event") == "privacy"]
    if privacy:
        st.subheader("客户端隐私预算")
        st.dataframe(pd.DataFrame(privacy).drop_duplicates("client_id", keep="last")[["client_id", "epsilon", "epsilon_cap", "steps"]], hide_index=True)
    if summary.get("devices"):
        st.subheader("设备参与情况")
        st.dataframe(pd.DataFrame(summary["devices"]), hide_index=True)
    assignments = read(selected / "assignments.json", {})
    if assignments:
        st.subheader("推理模型分配")
        st.dataframe(pd.DataFrame([{"客户端": k, "模型": v or "暂停：无满足预算的模型"}
                                   for k, v in assignments["assignments"].items()]), hide_index=True)
    registry = read(selected / "registry.json", {})
    if registry:
        st.subheader("模型质量与部署测量")
        st.dataframe(pd.DataFrame([{"模型": m["name"], "大小 KiB": m["artifact_bytes"] / 1024,
                                   "CPU p95 ms": m["p95_ms"], "验证 F1": m["validation"]["macro_f1"],
                                   "压缩倍数": m["compression_ratio"]} for m in registry["models"]]), hide_index=True)
    if summary.get("all_physical_leaves_verified") is False:
        st.warning("部分数据缺少官方物理叶片映射；已做重复图像审计，仍需保留叶片隔离局限。")


live()
