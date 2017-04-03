#!/usr/bin/python
import argparse
import logging
import poolvr.app


if __name__ == "__main__":
    FORMAT = '  POOLVR.PY 0.0.1  | %(asctime)s | %(name)s --- %(levelname)s *** %(message)s'
    parser = argparse.ArgumentParser()
    parser.add_argument("--novr", help="non-VR mode", action="store_true")
    parser.add_argument("--use_simple_ball_collisions", help="use simple ball collision model", action="store_true")
    parser.add_argument("--use_ode_physics", help="use ODE for physics simulation instead of the default event-based engine",
                        action="store_true")
    parser.add_argument('--multisample', help="enable multisampled VR rendering", action="store_true")
    parser.add_argument("-v", help="verbose logging", action="store_true")
    args = parser.parse_args()
    if args.v:
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FORMAT, level=logging.WARNING)
    if args.multisample:
        args.multisample = 1
    else:
        args.multisample = 0
    poolvr.app.main(novr=args.novr, use_simple_ball_collisions=args.use_simple_ball_collisions,
                    use_ode_physics=args.use_ode_physics,
                    multisample=args.multisample)
