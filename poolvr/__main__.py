from sys import exit
import argparse
import logging
_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = '%(name)s.%(funcName)s[%(levelname)s]: %(message)s'
_DEBUG_LOGGING_FORMAT = '%(asctime).19s [%(levelname)s]%(name)s.%(funcName)s (%(filename)s:%(lineno)d): %(message)s'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--verbose',
                        help="enable verbose logging",
                        action="store_true")
    parser.add_argument("--novr", help="non-VR mode", action="store_true")
    parser.add_argument("-a", "--msaa", metavar='<multisample level>',
                        help='enable multi-sampled anti-aliasing (disabled by default) at level A (1, 2, or 4)',
                        default=0)
    parser.add_argument('-o', "--ode",
                        help="use ODE for physics simulation instead of the default event-based engine",
                        action="store_true")
    parser.add_argument("-c", "--collision-model",
                        help="name of ball collision model to use (only applies to the event-based physics engine)",
                        default='simple')
    parser.add_argument('--bb-particles',
                        help='render balls using billboard particle shader instead of polygon meshes',
                        action='store_true')
    parser.add_argument('-s', '--sound-device',
                        metavar='<device ID>',
                        help="specify the sound device to use for output",
                        type=int)
    parser.add_argument('-l', '--list-sound-devices', help="list the available sound devices",
                        action="store_true")
    parser.add_argument('--cube-map', help='set cube map texture', default=None)
    parser.add_argument('--glyphs', help='render velocity and angular velocity glyphs',
                        action='store_true')
    parser.add_argument('--speed', help='time speed-up/slow-down factor (default is 1.0, normal speed)',
                        default=1.0, type=float)
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
        from .sound import list_sound_devices
        list_sound_devices()
        exit(0)
    if args.sound_device:
        import poolvr.sound
        poolvr.sound.set_output_sound_device(args.sound_device)
    import poolvr.app
    poolvr.app.main(novr=args.novr,
                    ball_collision_model=args.collision_model,
                    use_ode=args.ode,
                    multisample=args.msaa,
                    use_bb_particles=args.bb_particles,
                    cube_map=None,
                    speed=args.speed,
                    glyphs=args.glyphs)


if __name__ == "__main__":
    main()
