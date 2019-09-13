from pstats import Stats
import os.path
from os import listdir
from io import StringIO
import re
from datetime import datetime
import logging
_logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import numpy as np



_LOGGING_FORMAT = '### %(asctime).19s.%(msecs).3s [%(levelname)s] %(name)s.%(funcName)s (%(filename)s:%(lineno)d) ###\n%(message)s'
_here = os.path.dirname(__file__)
pstats_dir = os.path.relpath(os.path.join(_here, os.path.pardir, 'pstats'))
TIME_REX  = re.compile(r"(?P<dow>\w+)\s+(?P<month>\w+)\s+(?P<day>\d+)\s+(?P<H>\d\d):(?P<M>\d\d):(?P<S>\d\d)\s+(?P<year>\d+)")
STATS_REX = re.compile(r"(?P<ncalls>\d+)\s+(?P<tottime>\d*\.?\d*)\s+(?P<percall>\d*\.?\d*)\s+"
                       r"(?P<cumtime>\d*\.?\d*)\s+(?P<cumpercall>\d*\.?\d*)\s+(?P<filename>\w:[\\](\w*\.?\w*[\\])*\w+\.py)")
MONTH_NOS = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr':  4, 'May':  5, 'Jun':  6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


def plot_profiles(files, function=None,
                  field='cumtime'):
    vals = []
    plotted_files = []
    for fil in files:
        out = StringIO()
        stats = Stats(fil, stream=out)
        stats.print_stats(function)
        lines = [ln.strip() for ln in out.getvalue().split('\n')]
        dtime = None
        for ln in lines:
            if dtime is None:
                m = TIME_REX.match(ln)
                if m:
                    H, M, S = m.group('H'), m.group('M'), m.group('S')
                    H = int(H[1:]) if H[0] == '0' else int(H)
                    M = int(M[1:]) if M[0] == '0' else int(M)
                    S = int(S[1:]) if S[0] == '0' else int(S)
                    dtime = datetime(int(m.group('year')), MONTH_NOS[m.group('month')], int(m.group('day')), H, M, S)
                    plotted_files.append(fil)
                    continue
            m = STATS_REX.match(ln)
            if m:
                val = float(m.group(field))
                vals.append((dtime, val))
    if vals:
        vals, plotted_files = zip(*sorted(zip(vals, plotted_files)))
        w = max(len(f) for f in plotted_files)
        dtime0 = vals[0][0]
        _logger.info('''{field}:
{timings}'''.format(field=field,
                    timings='\n'.join(('{dt}  {filename:>%ds}:  {val}' %
                                       (w+2)).format(dt=dt, filename=fil, val=val)
                                      for fil, (dt, val) in zip(plotted_files, vals))))
        ts = np.array([(dt-dtime0).total_seconds() for dt, v in vals])
        plt.figure()
        plt.title('%s: %s' % (function, field))
        plt.plot(ts, np.array([ct for dt, ct in vals]), 'x')
        # plt.xticks(ts, [str(dt) for dt, v in vals], fontsize='x-small', rotation=75)
        plt.xticks(ts, plotted_files, fontsize='x-small', rotation=75)
        plt.ylabel(field)
        plt.show()


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', metavar="<directory>",
                        help="examine files in the specified directory",
                        default=None)
    parser.add_argument('-f', '--function', metavar="<function name>",
                        help="plot times for the specified function",
                        default="_determine_next_event")
    parser.add_argument('-r', '--rex', metavar="<regular expression pattern>",
                        help="only include files matching the regular expression",
                        default=r".*")
    parser.add_argument('--field', metavar="<profile field>",
                        help="statistic to plot (default: cumtime)",
                        default='cumtime')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format=_LOGGING_FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=_LOGGING_FORMAT, level=logging.INFO)
    if args.dir is None:
        args.dir = pstats_dir
    filename_rex = re.compile(args.rex.strip())
    files = [os.path.join(args.dir, f)
             for f in listdir(args.dir) if f.endswith('.pstats') and filename_rex.match(f)]
    plot_profiles(files, args.function, field=args.field)


if __name__ == "__main__":
    main()
