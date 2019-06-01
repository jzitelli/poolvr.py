from sys import exit
import argparse
import logging
_logger = logging.getLogger(__name__)
_LOGGING_FORMAT = '%(name)s.%(funcName)s[%(levelname)s]: %(message)s'
_DEBUG_LOGGING_FORMAT = '### %(asctime).19s.%(msecs).3s [%(levelname)s] %(name)s.%(funcName)s (%(filename)s:%(lineno)d) ###\n%(message)s'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--verbose',
                        help="enable verbose logging",
                        action="store_true")
    parser.add_argument("--novr",
                        help="non-VR mode",
                        action="store_true")
    parser.add_argument("-a", "--msaa",
                        metavar='<multisample level>', type=int,
                        help='enable multi-sampled anti-aliasing at specified level (must be a non-negative power of 2); default is 4',
                        default=4)
    parser.add_argument('--resolution',
                        help='OpenGL viewport resolution, e.g. 960x680',
                        default='960x680')
    parser.add_argument('--fullscreen',
                        help='create fullscreen window',
                        action='store_true')
    parser.add_argument('--bb-particles',
                        help='render balls using billboard particle shader instead of polygon meshes',
                        action='store_true')
    parser.add_argument('-o', "--ode",
                        help="use ODE for physics simulation instead of the default event-based physics engine",
                        action="store_true")
    parser.add_argument("-c", "--collision-model",
                        metavar='<name of collision model>',
                        help="set the ball-to-ball collision model to use (this parameter only applies to the event-based physics engine)",
                        default='simple')
    parser.add_argument('-q', '--use-quartic-solver',
                        help="solve for collision times using the internal quartic solver instead of numpy.roots",
                        action='store_true')
    parser.add_argument('-s', '--sound-device',
                        metavar='<device ID>',
                        help="enable sound using the specified device",
                        default=None)
    parser.add_argument('-l', '--list-sound-devices',
                        help="list the available sound devices",
                        action="store_true")
    parser.add_argument('--cube-map',
                        help='enable cube-mapped environmental texture',
                        action='store_true')
    parser.add_argument('--glyphs',
                        help='render velocity and angular velocity glyphs',
                        action='store_true')
    parser.add_argument('--speed',
                        metavar='<factor>',
                        help='time speed-up/slow-down factor (default is 1.0, normal speed)',
                        default=1.0, type=float)
    parser.add_argument('-r', '--realtime',
                        action='store_true',
                        help='enable the realtime version (intended for interactive usage) of the event-based physics engine')
    parser.add_argument('--collision_search_time_forward',
                        help='''time into the future in seconds to calculate events for
                        before yielding to render a new frame - using this option enables the realtime engine''')
    parser.add_argument('--collision_search_time_limit',
                        help='maximum time in seconds to spend calculating events before yielding to render a new frame - using this option enables the realtime engine')
    parser.add_argument('--balls-on-table',
                        help='comma-separated list of balls on table',
                        default=','.join(str(n) for n in range(16)))
    parser.add_argument('--render-method',
                        help='OpenGL rendering method/style to use, one of: "ega", "lambert", "billboards", "raycast"',
                        default='raycast')
    args = parser.parse_args()
    args.msaa = int(args.msaa)
    args.balls_on_table = [int(n) for n in args.balls_on_table.split(',')]
    args.resolution = [int(x) for x in args.resolution.split('x')]
    if args.collision_search_time_limit is not None:
        collision_search_time_limit = float(args.collision_search_time_limit)
    elif args.realtime:
        collision_search_time_limit = None
    else:
        collision_search_time_limit = None
    args.collision_search_time_limit = collision_search_time_limit
    if args.collision_search_time_forward is not None:
        collision_search_time_forward = float(args.collision_search_time_forward)
    elif args.realtime:
        collision_search_time_forward = 4.0/90
    else:
        collision_search_time_forward = None
    args.collision_search_time_forward = collision_search_time_forward
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
        start_sound(args.sound_device)
    import poolvr.app
    from poolvr.gl_techniques import LAMBERT_TECHNIQUE, EGA_TECHNIQUE
    poolvr.app.main(novr=args.novr,
                    ball_collision_model=args.collision_model,
                    use_ode=args.ode,
                    multisample=args.msaa,
                    use_bb_particles=args.bb_particles,
                    cube_map=args.cube_map,
                    speed=args.speed,
                    glyphs=args.glyphs,
                    balls_on_table=args.balls_on_table,
                    render_method=args.render_method,
                    use_quartic_solver=args.use_quartic_solver,
                    collision_search_time_forward=args.collision_search_time_forward,
                    collision_search_time_limit=args.collision_search_time_limit,
                    fullscreen=args.fullscreen,
                    window_size=args.resolution)


def start_sound(sound_device):
    try:
        sound_device = int(sound_device)
        try:
            import poolvr.sound
            try:
                poolvr.sound.set_output_sound_device(sound_device)
            except Exception as err:
                _logger.error('could not set output sound device:\n%s', err)
        except Exception as err:
            _logger.error('could not import poolvr.sound:\n%s', err)
    except Exception as err:
        _logger.error('could not parse parameter "--sound-device %s":\n%s', sound_device, err)


if __name__ == "__main__":
    main()
