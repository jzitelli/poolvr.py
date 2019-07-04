from pstats import Stats
import os.path
from os import listdir
from io import StringIO
import logging
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


_logger = logging.getLogger(__name__)
_here = os.path.dirname(__file__)
pstats_dir = os.path.join(_here, os.path.pardir, 'pstats')
TIME_REX = re.compile(r"(?P<dow>\w+)\s+(?P<month>\w+)\s+(?P<day>\d+)\s+(?P<H>\d\d):(?P<M>\d\d):(?P<S>\d\d)\s+(?P<year>\d+)")
STATS_REX = re.compile(r"(?P<ncalls>\d+)\s+(?P<tottime>\d*\.?\d*)\s+(?P<percall>\d*\.?\d*)\s+(?P<cumtime>\d*\.?\d*)\s+(?P<cumpercall>\d*\.?\d*)\s+(?P<filename>\w:[\\](\w*\.?\w*[\\])*\w+\.py)")
MONTH_NOS = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


files = [os.path.join(pstats_dir, f) for f in listdir(pstats_dir) if f.endswith('.pstats')]
cumtimes = []
dtimes = []
for fil in files:
    out = StringIO()
    stats = Stats(fil, stream=out)
    stats.print_stats('_determine_next_event')
    lines = [ln.strip() for ln in out.getvalue().split('\n')]
    for ln in lines:
        m = TIME_REX.match(ln)
        if m:
            H, M, S = m.group('H'), m.group('M'), m.group('S')
            H = int(H[1:]) if H[0] == '0' else int(H)
            M = int(M[1:]) if M[0] == '0' else int(M)
            S = int(S[1:]) if S[0] == '0' else int(S)
            dtimes.append(datetime(int(m.group('year')), MONTH_NOS[m.group('month')], int(m.group('day')), H, M, S))
            continue
        m = STATS_REX.match(ln)
        if m:
            cumtime = float(m.group('cumtime'))
            if cumtime < 0.1:
                cumtimes.append((dtimes[-1], cumtime))
if cumtimes:
    cumtimes.sort()
    # cumtimes = cumtimes[-90:]
    dtime0 = cumtimes[0][0]
    ts = np.array([(dt-dtime0).total_seconds() for dt, ct in cumtimes])
    ts, cumtimes = zip(*[(t, cumtimes[i])
                         for i, t in enumerate(ts)
                         if i == 0 or (t - ts[i-1]) > 400])
    ctimes = np.array([ct for dt, ct in cumtimes])
    plt.plot(ts, ctimes, 'x')
    plt.xticks(ts, [str(dt) for dt, ct in cumtimes], fontsize='xx-small', rotation=75)
    plt.show()
