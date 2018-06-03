from sys import exit
import argparse
import logging
_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = '%(name)s.%(funcName)s[%(levelname)s]: %(message)s'
_DEBUG_LOGGING_FORMAT = '%(asctime).19s [%(levelname)s]%(name)s.%(funcName)s:%(lineno)d: %(message)s'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--verbose',
                        help="enable verbose logging",
                        action="store_true")
    parser.add_argument("--novr", help="non-VR mode", action="store_true")
    parser.add_argument("-a", "--msaa", metavar='A',
                        help='enable multi-sampled anti-aliasing (disabled by default) at level A (1, 2, or 4)',
                        default=0)
    parser.add_argument("--use_simple_ball_collisions", help="use simple ball collision model",
                        action="store_true")
    parser.add_argument('-o', "--ode",
                        help="use ODE for physics simulation instead of the default event-based engine",
                        action="store_true")
    parser.add_argument('--bb_particles',
                        help='render balls using billboard particle shader instead of polygon meshes',
                        action='store_true')
    parser.add_argument('-s', '--sound_device', help="specify the sound device to use for output",
                        type=int)
    parser.add_argument('-l', '--list_sound_devices', help="list the available sound devices",
                        action="store_true")
    args = parser.parse_args()
    args.msaa = int(args.msaa)
    return args


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(format=_DEBUG_LOGGING_FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=_LOGGING_FORMAT, level=logging.WARNING)
    if args.list_sound_devices:
        import poolvr.sound
        poolvr.sound.list_sound_devices()
        exit(0)
    if args.sound_device:
        import poolvr.sound
        poolvr.sound.set_output_sound_device(args.sound_device)
    import poolvr.app
    poolvr.app.main(novr=args.novr,
                    use_simple_ball_collisions=args.use_simple_ball_collisions,
                    use_ode=args.ode,
                    multisample=args.msaa,
                    use_bb_particles=args.bb_particles)


if __name__ == "__main__":
    main()
