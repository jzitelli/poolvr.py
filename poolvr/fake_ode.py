from logging import getLogger
_logger = getLogger(__name__)


ode = None
def ode_or_fake_it(func):
    from functools import wraps
    @wraps(func)
    def try_import_ode(*args, **kwargs):
        global ode
        if ode is None:
            try:
                import ode
            except Exception as err:
                _logger.error(err)
                import fake_ode as ode
                import sys
                sys.modules['ode'] = ode
        func(*args, **kwargs)
    return try_import_ode


class Body(object):
    def __init__(self, world):
        pass
    def setMass(self, mass):
        pass
    def setPosition(self, *args, **kwargs):
        pass
    def setQuaternion(self, *args, **kwargs):
        pass
    def setLinearVel(self, *args, **kwargs):
        pass
    def setAngularVel(self, *args, **kwargs):
        pass


class Mass(object):
    def __init__(self):
        pass
    def setCylinderTotal(self, *args, **kwargs):
        pass


class World(object):
    pass
