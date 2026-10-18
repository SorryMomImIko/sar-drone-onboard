"""Монитор нагрузки. Пишет сводку в monitor_out.txt при получении SIGTERM.
Не трогает predict.py, запускается отдельным процессом."""
import signal, sys, time, glob, os

INTERVAL = 0.5
samples = {'total': [], 'per_core': [], 'freq': [], 'temp': [], 'memused': []}
running = True

def stop(*a):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def cpu_raw():
    res = []
    with open('/proc/stat') as f:
        for line in f:
            if not line.startswith('cpu'):
                break
            p = line.split()
            v = [int(x) for x in p[1:]]
            res.append((sum(v), v[3] + v[4]))   # total, idle+iowait
    return res

def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None

def temps():
    out = {}
    for z in sorted(glob.glob('/sys/class/thermal/thermal_zone*')):
        try:
            with open(z + '/type') as f:
                t = f.read().strip()
            v = read_int(z + '/temp')
            if v is not None:
                out[t] = v / 1000.0
        except Exception:
            pass
    return out

def mem_used_mb():
    total = avail = None
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                total = int(line.split()[1])
            elif line.startswith('MemAvailable:'):
                avail = int(line.split()[1])
            if total and avail:
                break
    return (total - avail) / 1024.0

prev = cpu_raw()
while running:
    time.sleep(INTERVAL)
    cur = cpu_raw()
    busy = []
    for (pt, pi), (ct, ci) in zip(prev, cur):
        dt, di = ct - pt, ci - pi
        busy.append(100.0 * (dt - di) / dt if dt else 0.0)
    prev = cur
    samples['total'].append(busy[0])
    samples['per_core'].append(busy[1:])
    f = [read_int(p) for p in sorted(glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq'))]
    samples['freq'].append([x for x in f if x])
    samples['temp'].append(temps())
    samples['memused'].append(mem_used_mb())

def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0

with open('monitor_out.txt', 'w') as out:
    n = len(samples['total'])
    out.write('=== НАГРУЗКА (замеров: %d, интервал %.1f c) ===\n' % (n, INTERVAL))
    if n:
        out.write('CPU всего:   средн %.1f%%   макс %.1f%%\n'
                  % (avg(samples['total']), max(samples['total'])))
        ncore = len(samples['per_core'][0])
        for i in range(ncore):
            col = [s[i] for s in samples['per_core']]
            out.write('  ядро %d:    средн %.1f%%   макс %.1f%%\n' % (i, avg(col), max(col)))
        fl = [max(s) for s in samples['freq'] if s]
        if fl:
            out.write('Частота:     средн %d МГц   макс %d МГц\n'
                      % (avg(fl) / 1000, max(fl) / 1000))
        keys = set()
        for s in samples['temp']:
            keys |= set(s)
        for k in sorted(keys):
            col = [s.get(k) for s in samples['temp'] if k in s]
            if col:
                out.write('Температура %s: средн %.1f°C   макс %.1f°C\n' % (k, avg(col), max(col)))
        out.write('ОЗУ занято:  средн %.0f МБ   макс %.0f МБ\n'
                  % (avg(samples['memused']), max(samples['memused'])))
