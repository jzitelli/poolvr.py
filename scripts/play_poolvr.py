#!/usr/bin/python
import argparse
import logging
import poolvr.app


if __name__ == "__main__":
    FORMAT = '  POOLVR.PY 0.0.1  | %(asctime)s | %(name)s --- %(levelname)s *** %(message)s'
    parser = argparse.ArgumentParser()
    parser.add_argument("--novr", help="non-VR mode", action="store_true")
    parser.add_argument("-v", help="verbose logging", action="store_true")
    args = parser.parse_args()
    if args.v:
        logging.basicConfig(format=FORMAT, level=logging.INFO)
    else:
        logging.basicConfig(format=FORMAT, level=logging.ERROR)
    poolvr.app.main(novr=args.novr)
