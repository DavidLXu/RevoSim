"""Functions usable directly as Isaac Lab ``ObservationTermCfg.func`` values."""


def tactile_pressure(env, sensor_name='tactile'):
    return env.scene[sensor_name].get_pressure()


def tactile_depth(env, sensor_name='tactile'):
    return env.scene[sensor_name].get_depth()


def tactile_marker_displacement(env, sensor_name='tactile'):
    return env.scene[sensor_name].get_marker_displacement()


def tactile_rgb(env, sensor_name='tactile'):
    return env.scene[sensor_name].get_rgb()
