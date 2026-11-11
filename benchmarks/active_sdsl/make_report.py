#!/usr/bin/env python3
"""Build the static HTML benchmark report from results/<host>/<variant>/*.json.

  python3 benchmarks/active_sdsl/make_report.py [--out benchmarks/active_sdsl/report/index.html]

Self-contained: plotly.js is embedded once, so the file works offline.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# Reference categorical palette (dataviz skill, light mode), fixed slot order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
HOST_COLOR = {"mac": SLOTS[0], "rpi": SLOTS[1]}
HOST_NAME = {"mac": "M1 Pro (Mac)", "rpi": "Raspberry Pi 4"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e2", "#fcfcfb"
MAP_ORDER = ["square", "symmetry_2", "plus", "apt"]

# Stage taxonomy for the per-step breakdown (8 slots max, fixed order).
STAGES = [
    ("C++ localize (fork-join)", [("localize", "localize_omp_forkjoin_3d")]),
    ("cleanup_SE2 (O(n²))", [("localize", "cleanup_SE2")]),
    ("fusion_2d", [("localize", "fusion_2d")]),
    ("connected components", [("localize", "connectedComponentsWithBeliefs"),
                              ("driver", "connectedComponentsWithBeliefs"),
                              ("simulate", "connectedComponentsWithBeliefs")]),
    ("room-distance BFS / EDT", [("simulate", "mover.min_distance_to_neighbor_rooms")]),
    ("move + collision checks", [("simulate", "mover._move_voxels_and_normalize_beliefs")]),
    ("entropy + ray casts + predicate", [("simulate", "entropy_SE2"), ("localize", "cast_rays"),
                                          ("localize", "Predicate_Fwd2D_Arr"),
                                          ("simulate", "mover.mark_voxels_visited")]),
]
OTHER = "other Python (residual)"


# ----------------------------------------------------------------------------- data
def load(host, variant="orig"):
    d = RES / host / variant
    out = {}
    if not d.exists():
        return out
    for f in d.glob("*.json"):
        if f.name.endswith(".failed.json"):
            continue
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    return out


def res(data, task):
    t = data.get(task)
    return t["result"] if t else None


def step_stage_times(step):
    st = step["stages"]
    vals = []
    for _, keys in STAGES:
        vals.append(sum(st.get(f"{c}/{n}", {}).get("s", 0.0) for c, n in keys))
    other = max(step["t_total"] - sum(vals), 0.0)
    return vals + [other]


def fmt_s(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if x >= 100:
        return f"{x:.0f} s"
    if x >= 1:
        return f"{x:.2f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.1f} ms"
    return f"{x * 1e6:.1f} µs"


def ratio(a, b):
    return a / b if a and b else float("nan")


# ----------------------------------------------------------------------------- plotting
def base_layout(fig, title, height=380, xlabel=None, ylabel=None, logy=False, logx=False):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=INK), x=0, xanchor="left"),
        height=height + 60, margin=dict(l=60, r=20, t=60, b=110),
        paper_bgcolor=SURF, plot_bgcolor=SURF,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK2),
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="left", x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID, title_text=xlabel,
                     type="log" if logx else None)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, title_text=ylabel,
                     type="log" if logy else None)
    return fig


def fig_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


# ----------------------------------------------------------------------------- sections
def sec_hosts(D):
    rows = []
    for h in ("mac", "rpi"):
        f = RES / h / "orig" / "host.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text())["host"]
        mem = info.get("mem_bytes")
        rows.append(f"<tr><td><span class='sw' style='background:{HOST_COLOR[h]}'></span>{HOST_NAME[h]}</td>"
                    f"<td>{html.escape(str(info.get('cpu_model')))}</td><td>{info.get('cpu_count')}</td>"
                    f"<td>{(mem or 0) / 2**30:.1f} GiB</td><td>{info.get('pointer_bits')}-bit</td>"
                    f"<td>{html.escape(info.get('platform', ''))}</td><td>{info.get('python')}</td>"
                    f"<td>{info.get('sdsl_version')} @ {info.get('git_head')}</td>"
                    f"<td>{'stubbed' if info.get('matplotlib_stubbed') else 'real'}</td></tr>")
    return ("<table class='t'><tr><th>host</th><th>CPU</th><th>cores</th><th>RAM</th><th>userland</th>"
            "<th>platform</th><th>Python</th><th>sdsl</th><th>matplotlib</th></tr>" + "".join(rows) + "</table>")


# The RPi ran a prefix of the M1's episodes; every cross-host view uses that common subset.
COMMON = {"episodes_deployed": (2, 10), "episodes_fixed": (1, 8)}  # (repeats, steps)


def common_episodes(data, task):
    r = res(data, task)
    if not r:
        return []
    reps, _ = COMMON.get(task, (10 ** 9, 10 ** 9))
    return [e for e in r["episodes"] if e["rep"] < reps]


def episode_steps(data, task="episodes_deployed"):
    _, steps = COMMON.get(task, (10 ** 9, 10 ** 9))
    return [(e, s) for e in common_episodes(data, task) for s in e["steps"] if s["step"] < steps]


def sec_step_overview(D):
    """Per-step wall time distribution, deployed mode, Mac vs RPi (orig + opt)."""
    fig = go.Figure()
    any_data = False
    for variant, dash in (("orig", None), ("opt", "opt")):
        for h in ("mac", "rpi"):
            steps = episode_steps(D[(h, variant)])
            if not steps:
                continue
            any_data = True
            xs = [e["map"] for e, s in steps]
            ys = [s["t_total"] for e, s in steps]
            fig.add_trace(go.Box(
                x=xs, y=ys, name=f"{HOST_NAME[h]} · {variant}", boxpoints="all", jitter=0.4, pointpos=0,
                marker=dict(color=HOST_COLOR[h], size=5, opacity=0.55 if variant == "orig" else 0.9,
                            symbol="circle" if variant == "orig" else "diamond"),
                line=dict(color=HOST_COLOR[h], width=1.5),
                fillcolor="rgba(0,0,0,0)" if variant == "orig" else None,
                hovertemplate="%{x}<br>%{y:.3f} s<extra>" + f"{HOST_NAME[h]} · {variant}" + "</extra>"))
    if not any_data:
        return "<p class='pending'>Episode data pending.</p>"
    fig.update_layout(boxmode="group")
    base_layout(fig, "Active-loop step wall time (deployed settings: depth 10, timeout 1 s)", 420,
                "map", "seconds per step (log)", logy=True)
    fig.update_xaxes(categoryorder="array", categoryarray=MAP_ORDER)
    return fig_html(fig)


def breakdown_table(D, task="episodes_deployed"):
    """Mean seconds per step by stage, for each (host, variant, map)."""
    rows = {}
    for (h, variant), data in D.items():
        steps = episode_steps(data, task)
        by_map = defaultdict(list)
        for e, s in steps:
            by_map[e["map"]].append(step_stage_times(s))
        for m, arr in by_map.items():
            rows[(h, variant, m)] = (np.mean(np.array(arr), axis=0), len(arr))
    return rows


def sec_breakdown(D, task="episodes_deployed", title_suffix="deployed"):
    rows = breakdown_table(D, task)
    if not rows:
        return "<p class='pending'>Episode data pending.</p>"
    labels = [n for n, _ in STAGES] + [OTHER]
    order = [(h, v, m) for m in MAP_ORDER for v in ("orig", "opt") for h in ("mac", "rpi") if (h, v, m) in rows]
    ylabels = [f"{m} · {HOST_NAME[h].split(' (')[0]}{' · opt' if v == 'opt' else ''}" for h, v, m in order]
    # one subplot per host so the bar lengths stay readable (RPi is ~10x slower)
    fig = make_subplots(rows=1, cols=2, subplot_titles=[HOST_NAME["mac"], HOST_NAME["rpi"]], horizontal_spacing=0.18)
    for ci, h in enumerate(("mac", "rpi")):
        sub = [(hh, v, m) for hh, v, m in order if hh == h]
        ylab = [f"{m}{' · opt' if v == 'opt' else ''}" for _, v, m in sub]
        for si, lab in enumerate(labels):
            vals = [rows[k][0][si] for k in sub]
            fig.add_trace(go.Bar(
                y=ylab, x=vals, orientation="h", name=lab, legendgroup=lab, showlegend=(ci == 0),
                marker=dict(color=SLOTS[si] if si < len(SLOTS) else MUTED, line=dict(color=SURF, width=1)),
                hovertemplate="%{y}<br>" + html.escape(lab) + ": %{x:.3f} s<extra></extra>"), row=1, col=ci + 1)
        fig.update_yaxes(autorange="reversed", row=1, col=ci + 1)
        fig.update_xaxes(title_text="mean seconds per step", row=1, col=ci + 1)
    fig.update_layout(barmode="stack", bargap=0.3)
    per_host = max(len([k for k in order if k[0] == h]) for h in ("mac", "rpi"))
    base_layout(fig, f"Where an active step spends its time ({title_suffix}) — mean per step; '· opt' rows = optimized examples",
                160 + 42 * max(per_host, 4))
    fig.update_yaxes(tickmode="linear", automargin=True)
    fig.update_layout(margin=dict(l=110, b=130))
    # table view (accessibility + exact numbers)
    th = "".join(f"<th>{html.escape(l)}</th>" for l in labels)
    trs = []
    for k in order:
        h, v, m = k
        vals, n = rows[k]
        tot = float(np.sum(vals))
        trs.append(f"<tr><td>{m}</td><td>{HOST_NAME[h]}</td><td>{v}</td><td>{n}</td><td><b>{fmt_s(tot)}</b></td>"
                   + "".join(f"<td>{fmt_s(x)}<span class='pct'> {100 * x / tot:.0f}%</span></td>" if tot else "<td>—</td>" for x in vals) + "</tr>")
    table = (f"<details><summary>Table view</summary><table class='t small'><tr><th>map</th><th>host</th><th>variant</th>"
             f"<th>steps</th><th>total</th>{th}</tr>{''.join(trs)}</table></details>")
    return fig_html(fig) + table


def sec_ratio(D):
    """RPi/Mac slowdown per stage (orig, fixed work = identical computation), pooled over common maps."""
    rows = breakdown_table({k: v for k, v in D.items() if k[1] == "orig"}, "episodes_fixed")
    labels = [n for n, _ in STAGES] + [OTHER]
    mac = [rows[k] for k in rows if k[0] == "mac"]
    rpi = [rows[k] for k in rows if k[0] == "rpi"]
    if not mac or not rpi:
        return "<p class='pending'>Needs both hosts.</p>", None
    common = sorted({k[2] for k in rows if k[0] == "mac"} & {k[2] for k in rows if k[0] == "rpi"})
    out = []
    for si, lab in enumerate(labels):
        m = sum(rows[("mac", "orig", mp)][0][si] for mp in common)
        r = sum(rows[("rpi", "orig", mp)][0][si] for mp in common)
        if m > 1e-5:
            out.append((lab, r / m, m, r))
    tm = sum(np.sum(rows[("mac", "orig", mp)][0]) for mp in common)
    tr = sum(np.sum(rows[("rpi", "orig", mp)][0]) for mp in common)
    fig = go.Figure(go.Bar(x=[o[1] for o in out], y=[o[0] for o in out], orientation="h",
                           marker=dict(color=SLOTS[1], line=dict(color=SURF, width=1)),
                           text=[f"{o[1]:.0f}×" for o in out], textposition="outside",
                           hovertemplate="%{y}: RPi is %{x:.1f}× slower<extra></extra>"))
    fig.add_vline(x=tr / tm, line=dict(color=INK2, dash="dot", width=1.5),
                  annotation_text=f"whole step {tr / tm:.1f}×", annotation_position="top")
    base_layout(fig, f"RPi slowdown vs M1 by stage — fixed-work episodes, identical computation (maps: {', '.join(common)})",
                340, "× slower", None)
    fig.update_yaxes(autorange="reversed")
    return fig_html(fig), tr / tm


def sec_localize_depth(D):
    maps = [m for m in MAP_ORDER]
    fig = make_subplots(rows=1, cols=len(maps), subplot_titles=maps, shared_yaxes=True, horizontal_spacing=0.03)
    any_data = False
    for h in ("mac", "rpi"):
        r = res(D[(h, "orig")], "localize_depth")
        if not r:
            continue
        for ci, m in enumerate(maps):
            recs = [x for x in r.get(m, []) if "depth" in x]
            by_pose = defaultdict(list)
            for x in recs:
                by_pose[tuple(round(v, 6) for v in x["pose"])].append(x)
            for pi, (pose, xs) in enumerate(by_pose.items()):
                any_data = True
                fig.add_trace(go.Scatter(
                    x=[x["depth"] for x in xs], y=[x["localize_s"] + x["cleanup_s"] for x in xs],
                    mode="lines+markers", name=HOST_NAME[h], legendgroup=h, showlegend=(ci == 0 and pi == 0),
                    line=dict(color=HOST_COLOR[h], width=2), marker=dict(size=7),
                    customdata=[[x["n_raw"], x["n_clean"], x["localize_s"], x["cleanup_s"]] for x in xs],
                    hovertemplate=(f"{HOST_NAME[h]} · pose {pi}<br>depth %{{x}}<br>total %{{y:.3f}} s<br>"
                                   "localize %{customdata[2]:.3f} s, cleanup %{customdata[3]:.3f} s<br>"
                                   "raw voxels %{customdata[0]}, after cleanup %{customdata[1]}<extra></extra>")),
                    row=1, col=ci + 1)
    if not any_data:
        return "<p class='pending'>pending</p>"
    base_layout(fig, "localize + cleanup_SE2 vs recursion depth (timeout disabled; one line per pose)", 380)
    fig.update_yaxes(type="log", gridcolor=GRID, title_text="seconds (log)", row=1, col=1)
    for ci in range(len(maps)):
        fig.update_xaxes(title_text="depth", row=1, col=ci + 1)
    return fig_html(fig)


def sec_threads(D):
    fig = go.Figure()
    any_data = False
    for h in ("mac", "rpi"):
        r = res(D[(h, "orig")], "threads")
        if not r:
            continue
        counts = [c for c in r["counts"] if "error" not in r["runs"].get(str(c), {"error": 1})]
        tot = {c: sum(x["s"] for x in r["runs"][str(c)]) for c in counts}
        if 1 not in tot:
            continue
        any_data = True
        fig.add_trace(go.Scatter(x=counts, y=[tot[1] / tot[c] for c in counts], mode="lines+markers",
                                 name=HOST_NAME[h], line=dict(color=HOST_COLOR[h], width=2), marker=dict(size=8),
                                 customdata=[tot[c] for c in counts],
                                 hovertemplate="%{x} threads: %{y:.2f}× (%{customdata:.2f} s)<extra>" + HOST_NAME[h] + "</extra>"))
    if not any_data:
        return "<p class='pending'>pending</p>"
    fig.add_trace(go.Scatter(x=[1, 8], y=[1, 8], mode="lines", name="ideal", line=dict(color=MUTED, dash="dot", width=1)))
    base_layout(fig, "OpenMP scaling of localize_omp_forkjoin (depth 10, sum over poses)", 340, "OMP threads", "speed-up ×")
    return fig_html(fig)


def sec_belief(D, native):
    kernels = ["cleanup_SE2", "connectedComponents(use3rdDim)", "fusion_2d(n x n)", "entropy_SE2"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=kernels, horizontal_spacing=0.06)
    any_data = False
    for h in ("mac", "rpi"):
        r = res(D[(h, "orig")], "belief_scaling")
        nat = native.get((h, "O3native")) or native.get((h, "Os"))
        for ci, k in enumerate(kernels):
            if r and k in r["kernels"]:
                pts = [x for x in r["kernels"][k] if "s" in x]
                any_data = True
                fig.add_trace(go.Scatter(x=[p["n"] for p in pts], y=[p["s"] for p in pts], mode="lines+markers",
                                         name=f"{HOST_NAME[h]} (Python call)", legendgroup=h + "py",
                                         showlegend=ci == 0, line=dict(color=HOST_COLOR[h], width=2), marker=dict(size=7),
                                         hovertemplate="n=%{x}: %{y:.4f} s<extra>" + HOST_NAME[h] + "</extra>"),
                              row=1, col=ci + 1)
            if nat and k in nat.get("belief_scaling", {}):
                pts = nat["belief_scaling"][k]
                fig.add_trace(go.Scatter(x=[p["n"] for p in pts], y=[p["s"] for p in pts], mode="lines",
                                         name=f"{HOST_NAME[h]} (native C++)", legendgroup=h + "c", showlegend=ci == 0,
                                         line=dict(color=HOST_COLOR[h], width=2, dash="dash"),
                                         hovertemplate="native n=%{x}: %{y:.4f} s<extra>" + HOST_NAME[h] + "</extra>"),
                              row=1, col=ci + 1)
    if not any_data:
        return "<p class='pending'>pending</p>"
    base_layout(fig, "Belief-layer kernels: time vs number of voxels (log–log)", 360)
    fig.update_xaxes(type="log", title_text="n voxels")
    fig.update_yaxes(type="log", gridcolor=GRID)
    fig.update_yaxes(title_text="seconds", row=1, col=1)
    return fig_html(fig)


def exps_table(D):
    trs = []
    for h in ("mac", "rpi"):
        r = res(D[(h, "orig")], "belief_scaling")
        if not r:
            continue
        for k, e in (r.get("exponents") or {}).items():
            last = [x for x in r["kernels"][k] if "s" in x][-1]
            trs.append(f"<tr><td>{HOST_NAME[h]}</td><td>{html.escape(k)}</td><td>n<sup>{e:.2f}</sup></td>"
                       f"<td>{last['n']}</td><td>{fmt_s(last['s'])}</td></tr>")
    return ("<table class='t small'><tr><th>host</th><th>kernel</th><th>fitted growth</th><th>largest n</th>"
            "<th>time at largest n</th></tr>" + "".join(trs) + "</table>") if trs else ""


def sec_micro(D):
    trs = []
    keys = None
    vals = {}
    for h in ("mac", "rpi"):
        r = res(D[(h, "orig")], "env_query")
        if not r:
            continue
        agg = defaultdict(list)
        for m, x in r.items():
            for k, v in x.items():
                if k.endswith("_us"):
                    agg[k].append(v)
        vals[h] = {k: statistics.mean(v) for k, v in agg.items()}
        keys = list(agg)
    po = {h: res(D[(h, "orig")], "py_overhead") for h in ("mac", "rpi")}
    rows = ""
    if keys:
        for k in keys:
            mv, rv = vals.get("mac", {}).get(k), vals.get("rpi", {}).get(k)
            rows += (f"<tr><td>{k.replace('_us', '')} (from Python, per call)</td><td>{mv:.2f} µs</td>"
                     f"<td>{(f'{rv:.2f} µs' if rv else '—')}</td><td>{(f'{rv / mv:.0f}×' if rv else '—')}</td></tr>")
    if po.get("mac"):
        for k, mv in po["mac"].items():
            rv = (po.get("rpi") or {}).get(k)
            unit = "ns/voxel" if "ns/voxel" in k else "ns"
            label = k.split(" [")[0]
            rows += (f"<tr><td>{html.escape(label)}</td><td>{mv:.0f} {unit}</td><td>{(f'{rv:.0f} {unit}' if rv else '—')}</td>"
                     f"<td>{(f'{rv / mv:.0f}×' if rv else '—')}</td></tr>")
    if not rows:
        return "<p class='pending'>pending</p>"
    return ("<table class='t small'><tr><th>operation</th><th>M1</th><th>RPi 4</th><th>RPi/M1</th></tr>"
            + rows + "</table>")


def sec_choose(D):
    trs = []
    maps = ["square", "symmetry_2", "plus"]
    fig = make_subplots(rows=1, cols=len(maps), subplot_titles=maps, shared_yaxes=True, horizontal_spacing=0.04)
    any_fig = False
    for h in ("mac", "rpi"):
        for v in ("orig", "opt"):
            r = res(D[(h, v)], "choose_movement")
            if not r:
                continue
            for m in MAP_ORDER:
                if m not in r:
                    continue
                x = r[m]
                trs.append(f"<tr><td>{HOST_NAME[h]}</td><td>{v}</td><td>{m}</td><td>{x['n_rooms']}</td>"
                           f"<td>{fmt_s(x['init_s'])}</td><td>{fmt_s(x['point_to_room_edge_distance_ms'] / 1e3)}</td>"
                           f"<td>{fmt_s(x['get_room_at_point_us'] / 1e6)}</td><td>{fmt_s(x['min_dist_bfs_ms'] / 1e3)}</td></tr>")
                sims = x.get("simulate_vs_n") or []
                if sims and m in maps:
                    any_fig = True
                    ci = maps.index(m)
                    fig.add_trace(go.Scatter(
                        x=[s["n"] for s in sims], y=[s["s"] for s in sims], mode="lines+markers",
                        name=f"{HOST_NAME[h]} · {'original' if v == 'orig' else 'optimized'}", legendgroup=f"{h}{v}",
                        showlegend=(ci == 0),
                        line=dict(color=HOST_COLOR[h], width=2, dash="solid" if v == "orig" else "dash"),
                        marker=dict(size=7, symbol="circle" if v == "orig" else "diamond"),
                        customdata=[s["n_sims"] for s in sims],
                        hovertemplate="n=%{x}: %{y:.4f} s (%{customdata} feasible moves)<extra>" + f"{h} {v} {m}" + "</extra>"),
                        row=1, col=ci + 1)
    out = ""
    if any_fig:
        base_layout(fig, "simulate_movements: seconds per call vs number of hypothesis voxels (16 candidate moves)", 400)
        fig.update_yaxes(type="log", gridcolor=GRID)
        fig.update_yaxes(title_text="seconds (log)", row=1, col=1)
        fig.update_xaxes(title_text="n voxels")
        out += fig_html(fig)
        out += ("<p class='note'>Cost depends on how many of the 16 moves pass the 5 % collision chance-constraint "
                "(hover shows it); with many scattered hypotheses almost every move is rejected and the call is cheap. "
                "Inputs: real localization outputs subsampled to n.</p>")
    if trs:
        out += ("<table class='t small'><tr><th>host</th><th>variant</th><th>map</th><th>rooms</th><th>mover init</th>"
                "<th>point_to_room_edge_distance</th><th>get_room_at_point</th><th>min-distance BFS</th></tr>"
                + "".join(trs) + "</table>")
    return out or "<p class='pending'>pending</p>"


def sec_profile(D):
    out = []
    for h in ("mac", "rpi"):
        r = res(D[(h, "orig")], "profile")
        if not r or not r.get("profiles"):
            continue
        agg = defaultdict(lambda: [0.0, 0])
        for m, p in r["profiles"].items():
            for row in p["top_tottime"]:
                a = agg[row["func"]]
                a[0] += row["tottime"]
                a[1] += row["ncalls"]
        tot = sum(v[0] for v in agg.values())
        top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:14]
        trs = "".join(f"<tr><td><code>{html.escape(f)}</code></td><td>{v[1]:,}</td><td>{fmt_s(v[0])}</td>"
                      f"<td>{100 * v[0] / tot:.1f}%</td></tr>" for f, v in top)
        calls = defaultdict(int)
        for e in r["episodes"]:
            for s in e["steps"]:
                for k, v in (s.get("env_calls") or {}).items():
                    calls[k] += v
        nsteps = sum(len(e["steps"]) for e in r["episodes"])
        callstr = ", ".join(f"{k}: {v / max(nsteps, 1):,.0f}/step" for k, v in calls.items())
        out.append(f"<div class='col'><h4>{HOST_NAME[h]} — cProfile, self time (all maps, {nsteps} steps)</h4>"
                   f"<table class='t small'><tr><th>function</th><th>calls</th><th>self time</th><th>share</th></tr>{trs}</table>"
                   f"<p class='note'>Python-level environment queries: {callstr}</p></div>")
    return "<div class='cols'>" + "".join(out) + "</div>" if out else "<p class='pending'>pending</p>"


def load_native():
    nat = {}
    for h in ("mac", "rpi"):
        d = RES / h / "native"
        if not d.exists():
            continue
        for f in d.glob("bench_*.json"):
            try:
                nat[(h, f.stem.replace("bench_", "").replace("_t1", "@1thread"))] = json.loads(f.read_text())
            except Exception:
                pass
    return nat


def sec_native_localize(native):
    rows, allrows = [], []
    for (h, variant), d in sorted(native.items()):
        if variant not in ("Os",):
            continue
        idx = defaultdict(int)
        for x in d["localize"]:
            x = dict(x, pose_idx=idx[x["map"]])
            idx[x["map"]] += 1
            allrows.append((h, x))
            if not x.get("aborted") and not x.get("quadratic_skipped") and x.get("t_orig", -1) > 0:
                rows.append((h, x))
    if not allrows:
        return "<p class='pending'>Native benchmark pending.</p>"
    fig = make_subplots(rows=1, cols=2, subplot_titles=[HOST_NAME["mac"], HOST_NAME["rpi"]], horizontal_spacing=0.16)
    for ci, h in enumerate(("mac", "rpi")):
        xs = [x for hh, x in rows if hh == h]
        if not xs:
            continue
        labels = [f"{x['map']} pose {x['pose_idx']} ({x['n_raw']:,} vox)" for x in xs]
        for si, (key, lab) in enumerate((("mirror_filter", "filter (parallel predicate)"),
                                         ("mirror_split", "split (serial, O(N²) reserve)"),
                                         ("mirror_verify", "verify (result discarded)"))):
            fig.add_trace(go.Bar(y=labels, x=[x[key] for x in xs], orientation="h", name=lab, legendgroup=lab,
                                 showlegend=ci == 0, marker=dict(color=SLOTS[[0, 1, 3][si]], line=dict(color=SURF, width=1)),
                                 hovertemplate="%{y}<br>" + lab + ": %{x:.4f} s<extra></extra>"), row=1, col=ci + 1)
        fig.update_yaxes(autorange="reversed", row=1, col=ci + 1)
        fig.update_xaxes(title_text="seconds (-Os build, all threads)", type="log", row=1, col=ci + 1)
    fig.update_layout(barmode="stack")
    base_layout(fig, "Inside localize_omp_forkjoin: time per phase (depth 10, no timeout)", 160 + 22 * max(len(rows) // 2, 6))
    fig.update_xaxes(type="log")
    # table with fixed variants
    trs = []
    for h, x in allrows:
        head = f"<tr><td>{HOST_NAME[h]}</td><td>{x['map']} pose {x['pose_idx']}</td>"
        if x.get("aborted"):
            trs.append(head + f"<td colspan='8'>aborted: more than 2,000,000 voxels entering a level "
                       f"(levels: {', '.join(f'{c:,}' for c in x['iter_counts'])}) — the pose is too ambiguous to finish safely</td></tr>")
            continue
        if x.get("quadratic_skipped"):
            trs.append(head + f"<td>{x['n_raw']:,}</td><td><i>not run</i> (a split of {x['max_split']:,} voxels; "
                       f"O(N²) → hours)</td><td>—</td><td>{fmt_s(x['t_fixed_reserve'])}</td><td>{fmt_s(x['t_fixed_all'])}</td>"
                       f"<td>—</td><td>{'✓' if x['outputs_identical'] else '✗'}</td><td>{fmt_s(x['t_cleanup_raw']) if x['t_cleanup_raw'] > 0 else 'not run (n too large)'}</td></tr>")
            continue
        trs.append(head + f"<td>{x['n_raw']:,}</td><td>{fmt_s(x['t_orig'])}</td>"
                   f"<td>{fmt_s(x['mirror_split'])} ({100 * x['mirror_split'] / max(x['t_mirror'], 1e-12):.0f}%)</td>"
                   f"<td>{fmt_s(x['t_fixed_reserve'])}</td><td>{fmt_s(x['t_fixed_all'])}</td>"
                   f"<td><b>{x['t_orig'] / max(x['t_fixed_all'], 1e-12):.1f}×</b></td>"
                   f"<td>{'✓' if x['outputs_identical'] else '✗'}</td><td>{fmt_s(x['t_cleanup_raw'])}</td></tr>")
    table = ("<details open><summary>Table: original vs. fixed variants (-Os build)</summary><table class='t small'>"
             "<tr><th>host</th><th>map</th><th>raw voxels</th><th>original</th><th>of which split</th>"
             "<th>+ pre-reserve</th><th>+ no final split / no verify</th><th>speed-up</th><th>same output</th>"
             "<th>cleanup_SE2 on raw</th></tr>" + "".join(trs) + "</table></details>")
    return fig_html(fig) + table


def sec_native_flags(native):
    """Compare build variants: total localize time and belief kernels at the largest common n."""
    hosts = sorted({h for h, _ in native})
    if not hosts:
        return "<p class='pending'>Native benchmark pending.</p>"
    order = ["Os", "O3", "O3native", "O3native_noerrno", "aarch64_O3native"]
    fig = make_subplots(rows=1, cols=len(hosts), subplot_titles=[HOST_NAME[h] for h in hosts], horizontal_spacing=0.1)
    trs = []
    for ci, h in enumerate(hosts):
        base = native.get((h, "Os"))
        if not base:
            continue

        def metrics(d):
            ok = [x for x in d["localize"] if not x.get("aborted")]
            orig = [x for x in ok if x.get("t_orig", -1) > 0]
            m = {"localize, original (sum)": sum(x["t_orig"] for x in orig),
                 "localize, fixed split (sum)": sum(x["t_fixed_all"] for x in ok),
                 "predicate filter phase (sum)": sum(x["fixedall_filter"] for x in ok)}
            for k, pts in d.get("belief_scaling", {}).items():
                if pts:
                    m[f"{k} @n={pts[-1]['n']}"] = (pts[-1]["n"], pts[-1]["s"])
            return m
        mb = metrics(base)
        variants = [v for v in order if (h, v) in native]
        for si, (mk, mv) in enumerate(mb.items()):
            ys = []
            for v in variants:
                mvv = metrics(native[(h, v)]).get(mk)
                if isinstance(mv, tuple):
                    ys.append(mv[1] / mvv[1] if mvv and mvv[0] == mv[0] else None)
                else:
                    ys.append(mv / mvv if mvv else None)
            fig.add_trace(go.Bar(x=variants, y=ys, name=mk, legendgroup=mk, showlegend=ci == 0,
                                 marker=dict(color=SLOTS[si % 8], line=dict(color=SURF, width=1)),
                                 hovertemplate="%{x}: %{y:.2f}× vs -Os<extra>" + html.escape(mk) + "</extra>"),
                          row=1, col=ci + 1)
            for v, y in zip(variants, ys):
                if y:
                    trs.append((h, mk, v, y))
        fig.add_hline(y=1, line=dict(color=INK2, width=1, dash="dot"), row=1, col=ci + 1)
    fig.update_layout(barmode="group")
    base_layout(fig, "Compiler-flag variants of the same C++ kernels: speed-up relative to -Os (what the extension uses)", 420)
    fig.update_yaxes(title_text="× faster than -Os", row=1, col=1)
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0, xanchor="left", yanchor="top"), margin=dict(b=130))
    return fig_html(fig)


def sec_alwaystrue(native, D):
    fig = go.Figure()
    any_data = False
    for h in ("mac", "rpi"):
        d = native.get((h, "Os"))
        if d:
            at = d["alwaystrue"]
            o = [x for x in at if x["t_orig"] >= 0]
            any_data = True
            fig.add_trace(go.Scatter(x=[8 ** (x["depth"] - 1) * 8 for x in o], y=[x["t_orig"] for x in o], mode="lines+markers",
                                     name=f"{HOST_NAME[h]} original", line=dict(color=HOST_COLOR[h], width=2), marker=dict(size=8),
                                     hovertemplate="%{x:,} voxels split: %{y:.4f} s<extra>original</extra>"))
            fig.add_trace(go.Scatter(x=[8 ** (x["depth"] - 1) * 8 for x in at], y=[x["t_fixed_reserve"] for x in at], mode="lines+markers",
                                     name=f"{HOST_NAME[h]} pre-reserved", line=dict(color=HOST_COLOR[h], width=2, dash="dash"),
                                     marker=dict(size=8, symbol="diamond"),
                                     hovertemplate="%{x:,} voxels split: %{y:.4f} s<extra>pre-reserved</extra>"))
    if not any_data:
        return "<p class='pending'>Native benchmark pending.</p>"
    base_layout(fig, "Voxel::split growth: AlwaysTrue predicate (pure subdivision), original vs pre-reserved output",
                380, "voxels produced by the last split (log)", "seconds (log)", logy=True, logx=True)
    return fig_html(fig)


def sec_deployed_quality(D):
    """How the 1 s timeout changes what the RPi computes (depth reached, voxel counts)."""
    trs = []
    for h in ("mac", "rpi"):
        steps = episode_steps(D[(h, "orig")])
        if not steps:
            continue
        by_map = defaultdict(list)
        for e, s in steps:
            by_map[e["map"]].append(s)
        for m in MAP_ORDER:
            ss = by_map.get(m)
            if not ss:
                continue
            its = [s["loc_log"]["iterations_run"] for s in ss if s.get("loc_log")]
            to = sum(1 for s in ss if s.get("loc_log") and s["loc_log"]["timeout_at"] is not None)
            tl = [s["t_localize"] for s in ss]
            nv = [s["n_vox"] for s in ss]
            trs.append(f"<tr><td>{HOST_NAME[h]}</td><td>{m}</td><td>{len(ss)}</td><td>{min(its) if its else '—'}–{max(its) if its else '—'}</td>"
                       f"<td>{to}/{len(ss)}</td><td>{fmt_s(statistics.median(tl))}</td><td>{fmt_s(max(tl))}</td>"
                       f"<td>{statistics.median(nv):.0f}</td></tr>")
    eps = []
    for h in ("mac", "rpi"):
        allep = common_episodes(D[(h, "orig")], "episodes_deployed")
        if not allep:
            continue
        for m in MAP_ORDER:
            es = [e for e in allep if e["map"] == m]
            if es:
                conv = sum(1 for e in es if e["converged"])
                wrong = sum(1 for e in es if e["converged"] is False)
                trunc = sum(1 for e in es if e["truncated"])
                eps.append(f"<tr><td>{HOST_NAME[h]}</td><td>{m}</td><td>{len(es)}</td><td>{conv}</td><td>{wrong}</td>"
                           f"<td>{len(es) - conv - wrong}</td><td>{trunc}</td>"
                           f"<td>{statistics.mean(min(len(e['steps']), COMMON['episodes_deployed'][1]) for e in es):.1f}</td>"
                           f"<td>{fmt_s(statistics.mean(sum(s['t_total'] for s in e['steps'] if s['step'] < COMMON['episodes_deployed'][1]) for e in es))}</td></tr>")
    if not trs:
        return "<p class='pending'>pending</p>"
    return ("<table class='t small'><tr><th>host</th><th>map</th><th>steps</th><th>recursion levels run</th>"
            "<th>steps that hit the timeout</th><th>median localize()</th><th>worst localize()</th><th>median voxels after cleanup</th></tr>"
            + "".join(trs) + "</table>"
            + "<h4>Episode outcomes (same seeds on both hosts; first 2 episodes, first 10 steps)</h4><table class='t small'><tr><th>host</th><th>map</th>"
              "<th>episodes</th><th>converged at the true pose</th><th>converged at a wrong pose</th><th>not converged</th>"
              "<th>truncated by budget</th><th>mean steps (≤10)</th><th>mean wall time of those steps</th></tr>" + "".join(eps) + "</table>"
            + "<p class='note'>\"Converged\" = the experiment's own test: the largest connected belief component holds ≥ 0.55 "
              "of the mass and its x–y box contains the true pose. It says nothing about how large that box is.</p>")


def rpi_thermal_samples():
    """All RPi thermal samples (original suite, optimized suite, native runs), sorted by time."""
    out = []
    for sub in ("orig", "opt", "native"):
        f = RES / "rpi" / sub / "events.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") == "thermal" and "temp_c" in e:
                out.append((e["t"], e["temp_c"], e.get("throttled"), sub))
    return sorted(out)


def sec_thermal():
    samples = rpi_thermal_samples()
    if not samples:
        return ""
    t0 = samples[0][0]
    xs, ys, cd = [], [], []
    prev_t = None
    for t, temp, thr, sub in samples:
        if prev_t is not None and t - prev_t > 900:  # break the line across idle gaps between runs
            xs.append(None); ys.append(None); cd.append(None)
        xs.append((t - t0) / 3600); ys.append(temp); cd.append([thr, sub])
        prev_t = t
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="SoC temperature", line=dict(color=SLOTS[1], width=2),
                               customdata=cd, connectgaps=False,
                               hovertemplate="%{x:.1f} h: %{y:.1f} °C · run=%{customdata[1]} · throttled=%{customdata[0]}<extra></extra>"))
    base_layout(fig, "RPi 4 SoC temperature over all runs (soft-throttling starts at 80 °C)", 280, "hours since the first run started", "°C")
    return fig_html(fig)


def _last_match(lines, pattern):
    import re
    hit = None
    for ln in lines:
        m = re.search(pattern, ln)
        if m:
            hit = m
    return hit


def sec_checks():
    """Build / test / equivalence status per host, parsed from phase2 logs."""
    import re
    rows = []
    for h, label in (("mac", "M1 Pro"), ("rpi", "RPi 4")):
        R = RES / h
        lines = (R / "phase2.log").read_text().splitlines() if (R / "phase2.log").exists() else []
        eq = _last_match(lines, r"equivalence exit=(\d+) : (.*)")
        app_c = _last_match(lines, r"demo app compile exit=(\d+)")
        app_r = _last_match(lines, r"demo app run exit=(\d+)")
        pt = (R / "pytest.log").read_text() if (R / "pytest.log").exists() else ""
        pm = re.findall(r"(\d+ passed[^\n]*?|\d+ failed[^\n]*?) in [\d.]+s", pt)
        nat = sorted(p.stem.replace("bench_", "") for p in (R / "native").glob("bench_*.json"))

        def ok(b, text):
            col = "var(--good)" if b else "var(--crit)"
            return f"<span style='color:{col};font-weight:600'>{'✓' if b else '✗'}</span> {html.escape(text)}"
        eq_txt = ok(eq and eq.group(1) == "0", eq.group(2).strip()) if eq else "<span class='pending'>not run</span>"
        if h == "rpi" and eq:
            eq_txt += " <span class='pct'>(quick: 1 map)</span>"
        app_txt = (ok(app_c.group(1) == "0", "compiles") + (" · " + ok(app_r.group(1) == "0", "runs") if app_r else "")) if app_c else "<span class='pending'>not run</span>"
        pt_txt = ok("failed" not in pm[-1], pm[-1]) if pm else "<span class='pending'>not run</span>"
        nat_txt = ok(True, ", ".join(nat)) if nat else ok(False, "no results")
        rows.append(f"<tr><td>{label}</td><td>{eq_txt}</td><td>{pt_txt}</td><td>{app_txt}</td><td>{nat_txt}</td></tr>")
    return ("<table class='t'><tr><th>host</th><th>equivalence test (orig vs opt examples)</th><th>pytest (installed sdsl)</th>"
            "<th>apps/demo_lidar_localization_pgm</th><th>native benchmark variants</th></tr>" + "".join(rows) + "</table>"
            "<p class='note'>The core library was not modified; the extension on both hosts is <code>pip3 install ./sdsl</code> at 328423d. "
            "The repository's top-level CMake build was also checked on the M1 (builds, 2 warnings).</p>")


# ----------------------------------------------------------------------------- page
CSS = """
:root{--bg:#f4f3f0;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e7e6e2;--accent:#2a78d6;--crit:#d03b3b;--good:#0ca30c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--line:#383835;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--line:#383835;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:28px;margin:8px 0 4px}h2{font-size:21px;margin:44px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}h4{font-size:14px;margin:14px 0 6px;color:var(--ink2)}
p,li{color:var(--ink2);max-width:80ch}.sub{color:var(--muted);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0;overflow-x:auto}
.plot{background:#fcfcfb;border-radius:8px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:26px;font-weight:650;color:var(--ink)}.kpi .l{font-size:12.5px;color:var(--muted)}
.notice{border-left:3px solid var(--accent);padding:2px 0 2px 14px;margin:12px 0}
.notice.crit{border-color:var(--crit)}.notice b{color:var(--ink)}
table.t{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}table.t th,table.t td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
table.t th{color:var(--muted);font-weight:600}table.small{font-size:12px}.pct{color:var(--muted);font-size:11px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
code{font-size:12px;background:rgba(127,127,127,.12);padding:1px 4px;border-radius:4px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:14px}.note{font-size:12.5px;color:var(--muted)}
.pending{color:var(--muted);font-style:italic}details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin:6px 0}
nav{font-size:13px;color:var(--muted)}nav a{color:var(--accent);text-decoration:none;margin-right:12px}
@media (max-width:600px){.cols{grid-template-columns:1fr}h1{font-size:23px}}
"""


def card(inner):
    return f"<div class='card'><div class='plot'>{inner}</div></div>"


def build(out_path, notes_path):
    D = {(h, v): load(h, v) for h in ("mac", "rpi") for v in ("orig", "opt")}
    native = load_native()
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    ratio_html, step_ratio = sec_ratio(D)

    def notes_block(key):
        items = notes.get(key, [])
        return "".join(f"<div class='notice{' crit' if it.get('crit') else ''}'>{it['html']}</div>" for it in items)

    kpis = notes.get("kpis", [])
    kpi_html = "<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='v'>{html.escape(k['v'])}</div><div class='l'>{html.escape(k['l'])}</div></div>" for k in kpis) + "</div>"

    body = f"""
<main>
<h1>Active SDSL on low-compute hardware: M1 Pro vs Raspberry Pi 4</h1>
<p class='sub'>Benchmark &amp; profiling report · generated {time.strftime('%Y-%m-%d %H:%M')} · harness <code>benchmarks/active_sdsl/</code></p>
<nav><a href='#notice'>Things to notice</a><a href='#e2e'>End-to-end</a><a href='#profile'>Profile</a><a href='#kernels'>Kernels</a>
<a href='#native'>Native C++</a><a href='#opt'>Optimizations applied</a><a href='#recs'>Core recommendations</a><a href='#method'>Method</a></nav>
{kpi_html}
<h2 id='notice'>Things to notice</h2>
{notes_block('notice')}
<h3>Machines</h3>{sec_hosts(D)}

<h2 id='e2e'>End-to-end active loop</h2>
<p>One <i>active step</i> = <code>demo_movement.localize()</code> (16 rays → fork-join localization → <code>cleanup_SE2</code> → <code>fusion_2d</code> → connected components)
+ the experiment driver's convergence check + <code>ChooseMovement.simulate_movements()</code> (16 candidate moves, collision chance-constraint, entropy, room-graph distance),
exactly as in <code>examples/experiment_movement.py</code>. Same seeds and start poses on both hosts.</p>
{card(sec_step_overview(D))}
{notes_block('e2e')}
{card(sec_breakdown(D))}
{card(ratio_html)}
<h3>What the 1 s timeout does on each machine</h3>
{notes_block('timeout')}
<div class='card'>{sec_deployed_quality(D)}</div>
<h3>Fixed-work episodes (timeout disabled → identical computation on both hosts)</h3>
{card(sec_breakdown(D, 'episodes_fixed', 'fixed work, timeout off'))}

<h2 id='profile'>Profile</h2>
{notes_block('profile')}
<div class='card'>{sec_profile(D)}</div>

<h2 id='kernels'>Kernel scaling</h2>
{notes_block('kernels')}
{card(sec_localize_depth(D))}
{card(sec_threads(D))}
{card(sec_belief(D, native))}
<div class='card'>{exps_table(D)}</div>
<h3>Per-call costs seen from Python</h3>
<div class='card'>{sec_micro(D)}</div>

<h2 id='native'>Native C++ kernels (core headers, unmodified)</h2>
{notes_block('native')}
{card(sec_native_localize(native))}
{card(sec_alwaystrue(native, D))}
{card(sec_native_flags(native))}

<h2 id='opt'>Optimizations applied (outside the core library)</h2>
{notes_block('opt')}
<h3>Build, test and equivalence checks</h3>
<div class='card'>{sec_checks()}</div>
{card(sec_choose(D))}

<h2 id='recs'>Recommendations for the core library (not applied)</h2>
{notes_block('recs')}

<h2 id='method'>Method &amp; caveats</h2>
{notes_block('method')}
{card(sec_thermal())}
</main>"""
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Active SDSL Benchmarks</title><style>{CSS}</style>"
            f"<script>{plotly.offline.get_plotlyjs()}</script></head><body>{body}</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "report" / "index.html"))
    ap.add_argument("--notes", default=str(HERE / "report" / "notes.json"))
    a = ap.parse_args()
    build(Path(a.out), Path(a.notes))
