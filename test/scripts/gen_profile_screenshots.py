import os
import logging
_logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to dumped cProfiles')
    parser.add_argument('--outdir', help='output directory', default='.')
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.path):
        _logger.error('%s is not a file or directory', args.path)
        sys.exit(1)
    return args


def main():
    args = parse_args()
    if os.path.isdir(args.path):
        filenames = os.listdir(args.path)
        if not filenames:
            return
        cwd = args.path
    else:
        filenames = [os.path.basename(args.path)]
        cwd = os.path.dirname(args.path)
    import subprocess
    proc = subprocess.Popen(['snakeviz', '-s', filenames[0]], cwd=cwd)
    try:
        from time import sleep
        from selenium import webdriver
        driver = webdriver.Firefox()
        try:
            for i_file, filename in enumerate(filenames):
                _logger.info('saving screenshot for profile "%s" (%d of %d)...', filename, i_file+1, len(filenames))
                driver.get("http://127.0.0.1:8080/snakeviz/C%3A%5CUsers%5Czemed%5CGitHub%5Cpoolvr.py%5Ctest%5Cpstats%5C{filename}".format(filename=filename))
                sleep(1)
                cutoff_elem = driver.find_element_by_id('sv-cutoff-select')
                if cutoff_elem:
                    pass
                svg_elem = driver.find_element_by_tag_name('svg')
                if svg_elem:
                    screenshot = os.path.join(args.outdir, '%s.png' % filename)
                    svg_elem.screenshot(screenshot)
                    _logger.info('...saved screenshot to "%s"', screenshot)
        finally:
            driver.quit()
    finally:
        proc.kill()


if __name__ == "__main__":
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    main()
