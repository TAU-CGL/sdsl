"""Live progress tracking via visdom (with a JSONL fallback that always works).

Panes, per host env ``sdsl_<host>``:
  status    HTML table: current task, overall %, elapsed, ETA, last results
  progress  overall % vs elapsed minutes
  eta       ETA (minutes) vs elapsed minutes
  thermal   CPU temperature / frequency (RPi) or load average (Mac)
  log       rolling log
  <task>    preliminary result plots pushed by individual tasks
Overview env ``sdsl_overview``: one status pane per host + a shared progress plot.
"""
from __future__ import annotations

import html
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from common import thermal_sample


def _fmt_dur(s):
    if s is None or not np.isfinite(s):
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


class Tracker:
    def __init__(self, host, tasks, server="http://localhost", port=8097, jsonl_path=None,
                 enabled=True):
        """tasks: list of (task_id, estimated_seconds)."""
        self.host = host
        self.env = f"sdsl_{host}"
        self.tasks = list(tasks)
        self.est = {t: e for t, e in self.tasks}
        self.done = {}             # task_id -> actual seconds
        self.status = {}           # task_id -> "done"/"failed"/"skipped"
        self.current = None
        self.cur_start = None
        self.cur_frac = 0.0
        self.cur_note = ""
        self.t0 = time.time()
        self.log_lines = deque(maxlen=300)
        self.prelim = {}           # short key -> text for the status pane
        self.jsonl = Path(jsonl_path) if jsonl_path else None
        if self.jsonl:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._last_push = 0.0
        self._last_thermal = 0.0
        self.vis = None
        if enabled:
            try:
                import visdom
                self.vis = visdom.Visdom(server=server, port=port, env=self.env,
                                         raise_exceptions=False, use_incoming_socket=False)
                if not self.vis.check_connection(timeout_seconds=5):
                    print(f"[tracker] visdom not reachable at {server}:{port}; JSONL only")
                    self.vis = None
            except Exception as e:  # pragma: no cover
                print(f"[tracker] visdom unavailable ({e!r}); JSONL only")
                self.vis = None
        if self.vis:
            for w in ("status", "progress", "eta", "thermal", "log"):
                self.vis.close(win=w, env=self.env)
        self.event("start", tasks=self.tasks)
        self.push(force=True)

    # ------------------------------------------------------------------ core
    def event(self, kind, **kw):
        rec = {"t": time.time(), "host": self.host, "kind": kind, **kw}
        if self.jsonl:
            with self.jsonl.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_lines.append(line)
        self.event("log", msg=msg)
        self.push()

    def _scale(self):
        """Ratio actual/estimated over finished tasks (ETA self-correction)."""
        act = sum(self.done.values())
        est = sum(self.est[t] for t in self.done)
        if est <= 0 or act <= 0:
            return 1.0
        return act / est

    def fraction(self):
        total = sum(self.est.values())
        if total <= 0:
            return 0.0
        d = sum(self.est[t] for t in self.done)
        if self.current:
            d += self.est[self.current] * self.cur_frac
        return min(1.0, d / total)

    def eta(self):
        scale = self._scale()
        remaining = sum(self.est[t] for t, _ in self.tasks if t not in self.done and t != self.current)
        if self.current:
            el = time.time() - self.cur_start
            if self.cur_frac > 0.05:
                cur_rem = el / self.cur_frac - el
            else:
                cur_rem = max(self.est[self.current] * scale - el, 0.0)
            remaining = remaining * scale + cur_rem
        else:
            remaining *= scale
        return remaining

    def begin(self, task_id):
        self.current = task_id
        self.cur_start = time.time()
        self.cur_frac = 0.0
        self.cur_note = ""
        idx = [t for t, _ in self.tasks].index(task_id) + 1
        self.log(f"▶ task {idx}/{len(self.tasks)}: {task_id} (est {_fmt_dur(self.est[task_id] * self._scale())})")
        self.event("begin", task=task_id)
        self.push(force=True)

    def sub(self, frac, note=""):
        self.cur_frac = max(0.0, min(1.0, frac))
        if note:
            self.cur_note = note
        self.push()

    def end(self, status="done", summary=""):
        t = self.current
        dt = time.time() - self.cur_start
        self.done[t] = dt
        self.status[t] = status
        self.current = None
        self.log(f"■ {t}: {status} in {_fmt_dur(dt)} {summary}")
        self.event("end", task=t, seconds=dt, status=status, summary=summary)
        if summary:
            self.prelim[t] = summary
        self.push(force=True)

    def result(self, key, text):
        """Preliminary result shown in the status pane."""
        self.prelim[key] = text
        self.event("prelim", key=key, text=text)
        self.push()

    # ----------------------------------------------------------- visdom panes
    def push(self, force=False):
        now = time.time()
        if not force and now - self._last_push < 2.0:
            return
        self._last_push = now
        frac = self.fraction()
        eta = self.eta()
        el = now - self.t0
        th = None
        if force or now - self._last_thermal > 10:
            th = thermal_sample()
            self._last_thermal = now
            self.event("thermal", **th)
        if not self.vis:
            return
        try:
            self._push_status(frac, eta, el)
            x = np.array([el / 60.0])
            self.vis.line(X=x, Y=np.array([100 * frac]), win="progress", update="append",
                          opts=dict(title=f"{self.host}: overall progress (%)", xlabel="elapsed [min]",
                                    ylabel="%", ytickmin=0, ytickmax=100))
            self.vis.line(X=x, Y=np.array([eta / 60.0]), win="eta", update="append",
                          opts=dict(title=f"{self.host}: ETA [min]", xlabel="elapsed [min]", ylabel="min"))
            self.vis.line(X=x, Y=np.array([100 * frac]), win="progress_all", env="sdsl_overview",
                          name=self.host, update="append",
                          opts=dict(title="overall progress (%) — all hosts", xlabel="elapsed [min]",
                                    ylabel="%", showlegend=True))
            if th is not None:
                if "temp_c" in th:
                    self.vis.line(X=x, Y=np.array([th["temp_c"]]), win="thermal", name="temp °C",
                                  update="append", opts=dict(title=f"{self.host}: SoC temperature / freq",
                                                             xlabel="elapsed [min]", showlegend=True))
                    if "freq_mhz" in th:
                        self.vis.line(X=x, Y=np.array([th["freq_mhz"] / 20.0]), win="thermal",
                                      name="freq MHz / 20", update="append")
                elif "load1" in th:
                    self.vis.line(X=x, Y=np.array([th["load1"]]), win="thermal", name="load1",
                                  update="append", opts=dict(title=f"{self.host}: 1-min load average",
                                                             xlabel="elapsed [min]", showlegend=True))
            self.vis.text("<pre style='font-size:11px'>" + html.escape("\n".join(list(self.log_lines)[-60:]))
                          + "</pre>", win="log", opts=dict(title=f"{self.host}: log"))
        except Exception as e:  # pragma: no cover
            print(f"[tracker] push failed: {e!r}")

    def _push_status(self, frac, eta, el):
        bar_w = 300
        rows = []
        for t, e in self.tasks:
            st = self.status.get(t, "running" if t == self.current else "pending")
            dur = self.done.get(t)
            if t == self.current:
                dur_s = f"{_fmt_dur(time.time() - self.cur_start)} ({100 * self.cur_frac:.0f}%) {html.escape(self.cur_note)}"
            else:
                dur_s = _fmt_dur(dur) if dur is not None else f"est {_fmt_dur(e * self._scale())}"
            color = {"done": "#2e7d32", "failed": "#c62828", "skipped": "#888", "running": "#1565c0"}.get(st, "#555")
            rows.append(f"<tr><td>{html.escape(t)}</td><td style='color:{color}'>{st}</td><td>{dur_s}</td></tr>")
        prelim = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>"
                         for k, v in list(self.prelim.items())[-14:])
        cur = html.escape(self.current or "—")
        body = f"""
<div style='font-family:sans-serif;font-size:12px'>
<h3 style='margin:2px'>{html.escape(self.host)} — {100 * frac:.1f}% — ETA {_fmt_dur(eta)}</h3>
<div style='width:{bar_w}px;background:#ddd;height:14px'><div style='width:{int(bar_w * frac)}px;background:#1565c0;height:14px'></div></div>
<p>elapsed {_fmt_dur(el)} · current: <b>{cur}</b> {html.escape(self.cur_note)} · ETA self-correction ×{self._scale():.2f}</p>
<table style='font-size:11px'>{''.join(rows)}</table>
<h4 style='margin:4px 0'>Preliminary results</h4><ul style='font-size:11px'>{prelim}</ul>
</div>"""
        self.vis.text(body, win="status", opts=dict(title=f"{self.host}: status"))
        self.vis.text(body, win=f"status_{self.host}", env="sdsl_overview",
                      opts=dict(title=f"{self.host}: status"))

    # ------------------------------------------------------- result plotting
    def plot_line(self, win, x, y, name, title, xlabel="", ylabel="", log_x=False, log_y=False,
                  overview=False):
        if not self.vis:
            return
        opts = dict(title=f"{self.host}: {title}", xlabel=xlabel, ylabel=ylabel, showlegend=True,
                    markers=True)
        if log_x:
            opts["xtype"] = "log"
        if log_y:
            opts["ytype"] = "log"
        try:
            self.vis.line(X=np.asarray(x, float), Y=np.asarray(y, float), win=win, name=name,
                          update="append" if self.vis.win_exists(win) else None, opts=opts)
            if overview:
                ov = dict(opts, title=f"{title} — all hosts")
                self.vis.line(X=np.asarray(x, float), Y=np.asarray(y, float), win=win, env="sdsl_overview",
                              name=f"{self.host}:{name}",
                              update="append" if self.vis.win_exists(win, env="sdsl_overview") else None,
                              opts=ov)
        except Exception as e:  # pragma: no cover
            print(f"[tracker] plot failed: {e!r}")

    def plot_bar(self, win, values, labels, title, legend=None, overview=False):
        if not self.vis:
            return
        try:
            v = np.asarray(values, float)
            opts = dict(title=f"{self.host}: {title}", rownames=list(labels))
            if legend:
                opts["legend"] = list(legend)
                opts["stacked"] = True
            self.vis.bar(X=v, win=win, opts=opts)
            if overview:
                self.vis.bar(X=v, win=f"{win}_{self.host}", env="sdsl_overview", opts=opts)
        except Exception as e:  # pragma: no cover
            print(f"[tracker] bar failed: {e!r}")

    def finish(self):
        self.current = None
        self.log(f"ALL DONE in {_fmt_dur(time.time() - self.t0)}")
        self.push(force=True)
        if self.vis:
            try:
                self.vis.save([self.env, "sdsl_overview"])
            except Exception:
                pass
