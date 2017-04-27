#!/usr/bin/python
import sys
import argparse
import logging


if __name__ == "__main__":
    FORMAT = '  POOLVR.PY 0.0.1  | %(asctime)s | %(name)s --- %(levelname)s *** %(message)s'
    parser = argparse.ArgumentParser()
    parser.add_argument("--novr", help="non-VR mode", action="store_true")
    parser.add_argument("--use_simple_ball_collisions", help="use simple ball collision model",
                        action="store_true")
    parser.add_argument("--use_ode_physics",
                        help="use ODE for physics simulation instead of the default event-based engine",
                        action="store_true")
    parser.add_argument('--multisample', help="set multisampling level for VR rendering",
                        type=int, default=0)
    parser.add_argument('--use_bb_particles',
                        help='render balls using billboard particle shader instead of polygon meshes',
                        action='store_true')
    parser.add_argument('--list_sound_devices', help="list the available sound devices",
                        action="store_true")
    parser.add_argument('--sound_device', help="specify the sound device to use for output",
                        type=int)
    parser.add_argument("-v", help="verbose logging", action="store_true")
    args = parser.parse_args()
    if args.v:
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FORMAT, level=logging.WARNING)
    if args.list_sound_devices:
        import poolvr.sound
        poolvr.sound.list_sound_devices()
        sys.exit(0)
    if args.sound_device:
        import poolvr.sound
        poolvr.sound.set_output_sound_device(args.sound_device)
    import poolvr.app
    poolvr.app.main(novr=args.novr,
                    use_simple_ball_collisions=args.use_simple_ball_collisions,
                    use_ode_physics=args.use_ode_physics,
                    multisample=args.multisample,
                    use_bb_particles=args.use_bb_particles)
