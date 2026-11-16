#!/usr/bin/env python3
"""q36_benchlog.py — everything worth knowing about a `q36-minipc bench` run, in ONE log file.

Called by q36-minipc (you do not normally run it by hand):

  header  write the machine / versions / resolved-knob preamble
  watch   background sampler: finds the engine process, records the environment IT ACTUALLY
          GOT (/proc/<pid>/environ), then every few seconds RSS, page faults, CPU, disk
          throughput, memory/IO pressure, clock speed
  finish  parse the engine's stderr, flag mismatches / thrashing, and append the summary,
          the sampler timeline, the generated text and the cleaned stderr to the log

Why it exists: field logs 6 and 7 turned out to be the same experiment because the knobs
never reached the engine, and nobody could tell without a 4-hour run. This log answers
"what did the engine actually run with, what did the machine look like, and where did the
time go" from one file. Every probe is best-effort: a missing tool is noted, never fatal.
Python 3.6+, standard library only.
"""
import argparse
import glob
import hashlib
import os
import re
import signal
import socket
import subprocess
import sys
import time

KEEP_ENV = re.compile(r"^(Q36_|COLI_|HOT_|GRIFFIN|TRUNK_|KV|SNAP|OMP_|GOMP_|RAM_GB|N_NEW|PIN|WIDE|PILOT|"
                      r"CAP_|SMOOTH|CONF_|USAGE_|MALLOC|LD_PRELOAD|WIN_|SERVE)")


# ---------------------------------------------------------------- small helpers
def sh(cmd, timeout=8):
    """Run a command, return its stdout ('' on any failure). Never raises."""
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        return r.stdout.decode("utf8", "replace").strip()
    except Exception:
        return ""


def rd(path, default=""):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf8", "replace").strip()
    except Exception:
        return default


def md5(path):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return "unreadable"


def section(title):
    return "\n== %s ==\n" % title


def kv_pairs(s):
    out = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def model_device(path):
    """(major, minor, partition name, parent disk name) of the block device holding `path`."""
    try:
        st = os.stat(path)
        major, minor = os.major(st.st_dev), os.minor(st.st_dev)
        real = os.path.realpath("/sys/dev/block/%d:%d" % (major, minor))
        name = os.path.basename(real)
        parent = os.path.basename(os.path.dirname(real)) if os.path.exists(os.path.join(real, "partition")) else name
        return major, minor, name, parent
    except Exception:
        return None, None, "?", "?"


# ---------------------------------------------------------------- header
def cmd_header(a):
    L = []
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    L.append("q36-minipc bench log — written by q36_benchlog.py")
    L.append(section("RUN").rstrip("\n"))
    L.append("started      : %s" % now)
    L.append("host / user  : %s / %s (uid %d)" % (socket.gethostname(), os.environ.get("USER", "?"), os.getuid()))
    L.append("command      : %s" % a.cmdline)
    L.append("model        : %s" % a.model)
    L.append("log file     : %s" % a.out)
    if a.meta:
        L.append(section("LAUNCHER DECISIONS").rstrip("\n"))
        for k, v in sorted(kv_pairs(a.meta).items()):
            L.append("  %-12s %s" % (k, v))

    L.append(section("ENVIRONMENT HANDED TO THE ENGINE (launcher's view)").rstrip("\n"))
    for k, v in sorted(kv_pairs(a.env).items()):
        L.append("  %s=%s" % (k, v))
    user = sorted("%s=%s" % (k, v) for k, v in os.environ.items() if k.startswith(("Q36_", "COLI_", "HOT_", "GRIFFIN", "TRUNK_")))
    L.append("  -- Q36_/COLI_/HOT_ variables set in the calling shell:")
    L.extend("  %s" % u for u in user) if user else L.append("  (none)")

    L.append(section("VERSIONS").rstrip("\n"))
    L.append("launcher     : %s  md5 %s" % (a.launcher, md5(a.launcher)))
    here = os.path.dirname(os.path.abspath(__file__))
    L.append("benchlog tool: %s  md5 %s" % (os.path.abspath(__file__), md5(os.path.abspath(__file__))))
    for t in ("project_q36.py", "hot_sort.py", "pack_trunk_i4.py", "make_layer_fracs.py"):
        for d in (here, os.path.expanduser("~/bin")):
            p = os.path.join(d, t)
            if os.path.exists(p):
                L.append("%-13s: %s  md5 %s" % (t[:13], p, md5(p)))
                break
    git = sh(["git", "-C", os.path.join(here, ".."), "log", "-1", "--format=%h %ad %s", "--date=short"])
    L.append("repo commit  : %s" % (git or "(launcher not run from a git checkout)"))
    eng = a.engine
    if os.path.exists(eng):
        st = os.stat(eng)
        try:
            blob = open(eng, "rb").read()
        except Exception:
            blob = b""
        marks = ["Q36_MAPI4_SCALAR", "Q36_MAPI4_IDOT", "Q36_EMBED_MAP", "Q36_WIN_MB", "HOT_FRAC", "GRIFFIN",
                 "TRUNK_MAP", "Q36_FLASH"]
        L.append("engine       : %s" % eng)
        L.append("  size %d bytes | built/modified %s | md5 %s" % (
            st.st_size, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)), md5(eng)))
        L.append("  knobs compiled in: " + ", ".join("%s=%s" % (m, "yes" if m.encode() in blob else "NO") for m in marks))
    else:
        L.append("engine       : %s  *** NOT FOUND ***" % eng)

    L.append(section("MACHINE").rstrip("\n"))
    L.append("kernel       : %s" % " ".join(os.uname()[:1] + os.uname()[2:3]))
    L.append("os           : %s" % (re.sub(r'^PRETTY_NAME="?|"?$', "", next((l for l in rd("/etc/os-release").splitlines() if l.startswith("PRETTY_NAME")), "?"))))
    lscpu = sh("lscpu")
    for key in ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "CPU max MHz", "CPU min MHz",
                "L2 cache", "L3 cache", "Virtualization", "Hypervisor vendor"):
        m = re.search(r"^%s:\s*(.+)$" % re.escape(key), lscpu, re.M)
        if m:
            L.append("  %-20s %s" % (key, m.group(1).strip()))
    flags = re.search(r"^Flags:\s*(.+)$", lscpu, re.M)
    if flags:
        fl = set(flags.group(1).split())
        L.append("  %-20s %s" % ("SIMD", " ".join("%s=%s" % (f, "yes" if f in fl else "no") for f in ("avx2", "fma", "avx512f", "avx_vnni"))))
    gov = rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "n/a")
    freqs = [int(rd(p, "0") or 0) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")]
    L.append("  %-20s %s (cur MHz now: %s)" % ("governor", gov, ", ".join(str(f // 1000) for f in freqs) or "n/a"))
    mem = rd("/proc/meminfo")
    for key in ("MemTotal", "MemAvailable", "Cached", "SwapTotal", "SwapFree", "HugePages_Total"):
        m = re.search(r"^%s:\s*(\d+)" % key, mem, re.M)
        if m:
            L.append("  %-20s %8.2f GiB" % (key, int(m.group(1)) / 1048576.0) if key != "HugePages_Total" else "  %-20s %s" % (key, m.group(1)))
    L.append("  %-20s %s | THP %s | ulimit -l %s" % ("vm", "swappiness " + rd("/proc/sys/vm/swappiness", "?"),
             rd("/sys/kernel/mm/transparent_hugepage/enabled", "?"), sh("bash -c 'ulimit -l'") or "?"))
    dmi = sh(["dmidecode", "-t", "memory"], timeout=10)
    if dmi:
        devs = []
        for blk in dmi.split("\n\n"):
            if "Memory Device" not in blk:
                continue
            f = dict(re.findall(r"^\s*([A-Za-z ]+):\s*(.+)$", blk, re.M))
            size = f.get("Size", "?")
            if "No Module" in size:
                continue
            devs.append("%s %s %s @ %s (configured %s)" % (f.get("Locator", "?"), size, f.get("Type", "?"),
                        f.get("Speed", "?"), f.get("Configured Memory Speed", f.get("Configured Clock Speed", "?"))))
        L.append("  %-20s %d populated DIMM(s)%s" % ("RAM modules", len(devs), " — 1 stick = single channel" if len(devs) == 1 else ""))
        L.extend("    " + d for d in devs)
    else:
        L.append("  %-20s (dmidecode unavailable — run as root to see stick count/speed)" % "RAM modules")

    L.append(section("STORAGE OF THE MODEL").rstrip("\n"))
    major, minor, part, disk = model_device(a.model)
    L.append("  device       : %s (%s:%s) on disk %s" % (part, major, minor, disk))
    q = "/sys/block/%s/queue/" % disk
    rot = rd(q + "rotational", "?")
    L.append("  rotational   : %s  (1 = spinning HDD, 0 = SSD/NVMe)" % rot)
    L.append("  read_ahead_kb: %s | scheduler: %s" % (rd(q + "read_ahead_kb", "?"), rd(q + "scheduler", "?")))
    lb = sh(["lsblk", "-d", "-o", "NAME,SIZE,ROTA,TRAN,MODEL", "/dev/" + disk])
    if lb:
        L.extend("  " + l for l in lb.splitlines())
    L.append("  filesystem   : %s" % (sh(["df", "-hT", a.model]).replace("\n", "\n                 ")))

    L.append(section("MODEL DIRECTORY").rstrip("\n"))
    try:
        for name in sorted(os.listdir(a.model)):
            p = os.path.join(a.model, name)
            if os.path.isfile(p):
                L.append("  %-34s %14d bytes" % (name, os.path.getsize(p)))
    except Exception as e:
        L.append("  (cannot list: %s)" % e)
    meta = rd(os.path.join(a.model, "qwen36_meta.json"))
    if meta:
        L.append("  qwen36_meta.json: " + re.sub(r"\s+", " ", meta)[:600])
    hl = os.path.join(a.model, "hot_layers.json")
    L.append("  hot_layers.json : %s" % (("present md5 " + md5(hl)) if os.path.exists(hl) else "absent"))
    man = os.path.join(a.model, "trunk_i4_manifest.json")
    L.append("  trunk manifest  : %s" % (re.sub(r"\s+", " ", rd(man))[:300] if os.path.exists(man) else "absent"))

    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- watch (background sampler)
def proc_args(pid):
    try:
        raw = open("/proc/%s/cmdline" % pid, "rb").read().split(b"\0")
        return [x.decode("utf8", "replace") for x in raw if x]
    except Exception:
        return []


def find_engine(engine_real, me):
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == me:
            continue
        a = proc_args(p)
        if not a or os.path.basename(a[0]) == "time":
            continue
        try:
            if os.path.realpath(a[0]) == engine_real:
                return int(p)
            if len(a) > 1 and os.path.basename(a[0]) in ("bash", "sh", "dash", "python3", "python") \
                    and os.path.realpath(a[1]) == engine_real:
                return int(p)
        except Exception:
            pass
    return None


def pstat(pid):
    s = rd("/proc/%d/stat" % pid)
    if ")" not in s:
        return None
    t = s.rsplit(")", 1)[1].split()
    return {"state": t[0], "minflt": int(t[7]), "majflt": int(t[9]), "cpu": int(t[11]) + int(t[12]), "thr": int(t[17])}


def status_kb(pid, key):
    m = re.search(r"^%s:\s*(\d+)" % key, rd("/proc/%d/status" % pid), re.M)
    return int(m.group(1)) if m else 0


def cpu_totals():
    t = rd("/proc/stat").splitlines()[0].split()[1:9]
    v = [int(x) for x in t]
    return sum(v), v[3] + v[4], v[4]          # total, idle+iowait, iowait


def disk_sectors(major, minor):
    for line in rd("/proc/diskstats").splitlines():
        f = line.split()
        if len(f) > 5 and f[0] == str(major) and f[1] == str(minor):
            return int(f[5])
    return 0


def psi(kind):
    m = re.search(r"^some avg10=([\d.]+)", rd("/proc/pressure/" + kind), re.M)
    return m.group(1) if m else "-"


def cmd_watch(a):
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    engine_real = os.path.realpath(a.engine)
    out = open(a.out, "w")
    major, minor, _, _ = model_device(a.model)
    me, t_wait = os.getpid(), time.time()
    pid = None
    while not stop and time.time() - t_wait < a.wait_start:
        pid = find_engine(engine_real, me)
        if pid:
            break
        time.sleep(0.5)
    if not pid:
        out.write("NOTE engine process never appeared within %ds\n" % a.wait_start)
        out.close()
        return
    t0 = time.time()
    try:
        env = open("/proc/%d/environ" % pid, "rb").read().split(b"\0")
        for e in sorted(x.decode("utf8", "replace") for x in env if x):
            if KEEP_ENV.match(e):
                out.write("ENV %s\n" % e)
    except Exception as e:
        out.write("NOTE cannot read engine environ: %s\n" % e)
    out.write("ARGV %s\n" % " ".join(proc_args(pid)))
    out.write("COLS t_s state thr rss_mb anon_mb file_mb majflt_s cpu_pct sys_busy_pct iowait_pct "
              "disk_mb_s avail_mb cached_mb swap_mb mem_psi io_psi mhz\n")
    out.flush()
    prev = (time.time(), pstat(pid), cpu_totals(), disk_sectors(major, minor))
    last = None
    while not stop:
        # sample on the interval, but notice exit within a second
        due = time.time() + a.interval
        while time.time() < due and not stop and os.path.exists("/proc/%d" % pid):
            time.sleep(min(1.0, max(0.05, due - time.time())))
        alive = os.path.exists("/proc/%d" % pid) and (pstat(pid) or {}).get("state") != "Z"
        ps = pstat(pid) if alive else None
        now = time.time()
        if ps is None:
            out.write("EXIT engine process ended at t=%.0fs (last sample above)\n" % (now - t0))
            break
        dt = max(now - prev[0], 1e-6)
        tot, idle, iow = cpu_totals()
        dtot = max(tot - prev[2][0], 1)
        mem = rd("/proc/meminfo")
        g = lambda k: int((re.search(r"^%s:\s*(\d+)" % k, mem, re.M) or [0, "0"])[1]) if re.search(r"^%s:\s*(\d+)" % k, mem, re.M) else 0
        fr = [int(rd(p, "0") or 0) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")]
        sec = disk_sectors(major, minor)
        row = ("%6.0f %s %3d %7.0f %7.0f %7.0f %8.1f %7.0f %6.0f %6.0f %8.1f %7.0f %7.0f %6.0f %6s %6s %5.0f" % (
            now - t0, ps["state"], ps["thr"], status_kb(pid, "VmRSS") / 1024.0, status_kb(pid, "RssAnon") / 1024.0,
            status_kb(pid, "RssFile") / 1024.0, (ps["majflt"] - prev[1]["majflt"]) / dt,
            100.0 * (ps["cpu"] - prev[1]["cpu"]) / (os.sysconf("SC_CLK_TCK") * dt),
            100.0 * (1 - float(idle - prev[2][1]) / dtot), 100.0 * float(iow - prev[2][2]) / dtot,
            (sec - prev[3]) * 512 / 1e6 / dt, g("MemAvailable") / 1024.0, g("Cached") / 1024.0,
            (g("SwapTotal") - g("SwapFree")) / 1024.0, psi("memory"), psi("io"),
            (sum(fr) / len(fr) / 1000.0) if fr else 0))
        out.write(row + "\n")
        out.flush()
        prev = (now, ps, (tot, idle, iow), sec)
        last = ps
    out.close()


# ---------------------------------------------------------------- finish (parse + assemble)
def first(rx, text, flags=re.M):
    m = re.search(rx, text, flags)
    return m.groups() if m else None


def cmd_finish(a):
    err = open(a.err, "rb").read().decode("utf8", "replace") if os.path.exists(a.err) else ""
    err = err.replace("\r", "\n")
    outtxt = open(a.bench_out, "rb").read().decode("utf8", "replace") if os.path.exists(a.bench_out) else ""
    tl = open(a.timeline).read() if os.path.exists(a.timeline) else ""
    eng_env = {}
    for line in tl.splitlines():
        if line.startswith("ENV "):
            k, _, v = line[4:].partition("=")
            eng_env[k] = v
    asked = kv_pairs(a.env)

    S = []
    S.append(section("SUMMARY").rstrip("\n"))
    S.append("exit status   : %s%s" % (a.rc, "" if str(a.rc) == "0" else "   *** ENGINE FAILED OR WAS INTERRUPTED ***"))
    S.append("wall time     : %s s (%s)" % (a.seconds, "%dh%02dm" % (int(a.seconds) // 3600, int(a.seconds) % 3600 // 60)))
    g = first(r"^== (qwen36 .*?) ==\s*$", err)
    S.append("engine banner : %s" % (g[0] if g else "(not found)"))
    g = first(r"resident weights loaded in ([\d.]+)s \| RSS after load: ([\d.]+) GB", err)
    if g:
        S.append("load          : %s s, RSS after load %s GB" % g)
    g = first(r"^TTFT: ([\d.]+) s", err)
    S.append("TTFT          : %s" % (("%s s (time to first token, includes prefill)" % g[0]) if g else "(not reached)"))
    g = first(r"\[timers\] decode: (\d+) tokens", err)
    tokens = int(g[0]) if g else 0
    totals = first(r"\[timers\]\s+TOTAL\s+([\d.]+) ms\s+([\d.]+) ms/token", err)
    if totals:
        per = float(totals[1]) / 1000.0
        S.append("DECODE        : %.2f s/token  (%.4f tok/s) over %d tokens   <<< the number that matters" % (per, 1.0 / per if per else 0, tokens))
    else:
        S.append("DECODE        : (no timers — did COLI_TIMERS=1 reach the engine?)")
    g = first(r"^Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)", err)
    if g:
        S.append("speed (incl. prefill): %s tok/s  (%s s for %s tokens)" % g)
    g = first(r"^PEAK RSS: ([\d.]+) GB", err)
    peak = float(g[0]) if g else 0.0
    S.append("peak RSS      : %s" % (("%.2f GB" % peak) if g else "(not found)"))

    S.append("")
    S.append("-- phase timers (per decoded token) --")
    seen_ph = set()
    for ph, ms in re.findall(r"\[timers\]\s+(\w+)\s+[\d.]+ ms\s+([\d.]+) ms/token", err):
        if ph not in seen_ph:
            seen_ph.add(ph)
            S.append("  %-10s %10.2f s/token" % (ph, float(ms) / 1000.0))
    g = first(r"\[timers\] prefill: (\d+) tokens (.*)$", err)
    if g:
        S.append("  prefill: %s tokens %s" % g)

    S.append("")
    S.append("-- caches / paging --")
    g = first(r"\[qwen36 win\] hits (\d+) misses (\d+) \(([\d.]+)% hit-rate\), budget (\d+) MB", err)
    S.append("window cache  : %s" % (("hits %s misses %s (%s%% hit rate) budget %s MB" % g) if g else "not active (no [qwen36 win] line)"))
    for rx, label in ((r"^.*\[warm\].*$", "warm"), (r"^.*\[mapi4\].*$", "kernel"), (r"^.*[Ee]mbed.?[Mm]ap.*$", "embed-map"),
                      (r"^.*HOT_LAYERS.*$", "hot layers"), (r"^.*GRIFFIN.*$", "griffin")):
        shown = []
        for line in re.findall(rx, err, re.M):
            line = line.strip()[:170]
            if line not in shown and len(shown) < 3:
                shown.append(line)
                S.append("%-13s : %s" % (label, line))
    inputs = first(r"File system inputs:\s*(\d+)", err)
    majf = first(r"Major \(requiring I/O\) page faults:\s*(\d+)", err)
    elapsed = first(r"Elapsed \(wall clock\) time.*: (\S+)", err)
    cpu = first(r"Percent of CPU this job got:\s*(\d+)%", err)
    mx = first(r"Maximum resident set size \(kbytes\):\s*(\d+)", err)
    read_gb = int(inputs[0]) * 512 / 1e9 if inputs else None
    if read_gb is not None:
        mbs = read_gb * 1000.0 / max(float(a.seconds), 1.0)
        S.append("disk read     : %.1f GB total (%.1f MB/s average)   major page faults: %s" % (read_gb, mbs, majf[0] if majf else "?"))
    if cpu:
        S.append("CPU used      : %s%% of one core (400%% = all 4 cores busy; low = waiting on disk)" % cpu[0])
    if mx:
        S.append("max RSS (time): %.2f GiB" % (int(mx[0]) / 1048576.0))

    S.append("")
    S.append("-- verdicts --")
    mism = []
    for k, v in sorted(asked.items()):
        if eng_env and eng_env.get(k) != v:
            mism.append("launcher passed %s=%s but the engine has %s" % (k, v, ("%s=%s" % (k, eng_env[k])) if k in eng_env else "NO such variable"))
    if not eng_env:
        S.append("  ? could not read the engine's environment (sampler never saw the process)")
    elif mism:
        S.append("  !! KNOB MISMATCH — settings did not reach the engine:")
        S.extend("       " + m for m in mism)
    else:
        S.append("  ok engine environment matches what the launcher passed (%d variables checked)" % len(asked))
    if "Q36_WIN_MB" not in asked and re.search(r"\[qwen36 win\]", err):
        S.append("  !! window cache is active although the launcher did not request it")
    if read_gb is not None and peak:
        if read_gb > 2 * max(peak, 1.0):
            S.append("  !! I/O: read %.0f GB for a %.1f GB working set — the page cache is THRASHING (weights re-read every token)" % (read_gb, peak))
        else:
            S.append("  ok I/O: read %.1f GB for a %.1f GB working set (no thrash signature)" % (read_gb, peak))
    memtot = re.search(r"MemTotal\s+([\d.]+)", open(a.out).read() if os.path.exists(a.out) else "")
    if memtot and peak:
        S.append("  %s headroom: peak RSS %.2f GB of %.2f GiB RAM (%.0f%% — above ~90%% leaves the page cache nothing)" % (
            "!!" if peak / float(memtot.group(1)) > 0.9 else "ok", peak, float(memtot.group(1)), 100 * peak / float(memtot.group(1))))

    S.append(section("ENGINE ENVIRONMENT (as the running engine saw it)").rstrip("\n"))
    S.extend("  %s=%s" % kv for kv in sorted(eng_env.items())) if eng_env else S.append("  (not captured)")
    S.append(section("TIMELINE (sampled every few seconds)").rstrip("\n"))
    rows = [l for l in tl.splitlines() if not l.startswith("ENV ")]
    S.extend(rows) if rows else S.append("  (none)")
    S.append(section("GENERATED TEXT (first 800 chars — sanity check for quality)").rstrip("\n"))
    S.append(outtxt[:800] if outtxt.strip() else "(empty)")
    S.append(section("ENGINE STDERR (cleaned; \\r -> newline)").rstrip("\n"))
    lines = err.splitlines()
    if len(lines) > 3500:
        lines = lines[:1200] + ["... [%d lines omitted] ..." % (len(lines) - 3200)] + lines[-2000:]
    S.extend(lines)

    with open(a.out, "a") as f:
        f.write("\n".join(S) + "\n")
    if not a.keep_timeline:
        try:
            os.remove(a.timeline)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="sub", required=True)
    h = sub.add_parser("header")
    h.add_argument("--out", required=True); h.add_argument("--model", required=True)
    h.add_argument("--engine", required=True); h.add_argument("--launcher", default="")
    h.add_argument("--env", default=""); h.add_argument("--cmdline", default=""); h.add_argument("--meta", default="")
    w = sub.add_parser("watch")
    w.add_argument("--engine", required=True); w.add_argument("--out", required=True)
    w.add_argument("--model", required=True); w.add_argument("--interval", type=float, default=30.0)
    w.add_argument("--wait-start", type=int, default=600)
    f = sub.add_parser("finish")
    f.add_argument("--out", required=True); f.add_argument("--err", required=True)
    f.add_argument("--bench-out", default="/dev/null"); f.add_argument("--timeline", required=True)
    f.add_argument("--rc", default="0"); f.add_argument("--seconds", default="0"); f.add_argument("--env", default="")
    f.add_argument("--keep-timeline", action="store_true")
    a = ap.parse_args()
    {"header": cmd_header, "watch": cmd_watch, "finish": cmd_finish}[a.sub](a)


if __name__ == "__main__":
    main()
