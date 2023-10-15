import logging
_logger = logging.getLogger(__name__)    
import json
import poolvr.physics
from poolvr.physics import PoolPhysics
from poolvr.table import PoolTable
from poolvr.physics.events import PhysicsEvent
import numpy as np


def pool_table():
    return PoolTable(ball_radius=PhysicsEvent.ball_radius)


def pool_physics(pool_table, ball_collision_model='fsimulated'):
    return PoolPhysics(initial_positions=pool_table.calc_racked_positions(),
                       ball_collision_model=ball_collision_model)


physics = pool_physics(pool_table())


def lambda_handler(event, context):
    # physics.reset(balls_on_table=[0])
    # ball_positions = physics.eval_positions(0.0)
    # r_c = ball_positions[0]
    # r_c[2] += physics.ball_radius
    # V = np.array((0, 0, -0.6), dtype=np.float64)
    # M = 0.54
    # events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    # print(event['balls_on_table'])

    print('balls_on_table: %s' % event['balls_on_table'])

    physics.reset(balls_on_table=event['balls_on_table'])
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    V = np.array((-0.01, 0.0, -1.6), dtype=np.float64)
    M = 0.54

    print('striking ball 0...')
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    
    print('strike on 0 resulted in %d events' % len(events))
    print(PhysicsEvent.events_str(events[:10]))
    
    # _logger.info('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
    #              PhysicsEvent.events_str(events))

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


lambda_handler({'balls_on_table': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]},
                None)
