#!/usr/bin/python
import argparse
import logging
import poolvr

if __name__ == "__main__":
    FORMAT = '  POOLVR 0.0a  | %(asctime)s | %(name)s --- %(levelname)s *** %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--novr", help="non-VR mode", action="store_true")
    args = parser.parse_args()
    poolvr.main(novr=args.novr)
