"""Data access remains lazy even through ``sensor.data``."""


class TactileSensorData:
    def __init__(self, sensor):
        self._sensor = sensor

    @property
    def pressure(self):
        return self._sensor.get_pressure()

    @property
    def depth(self):
        return self._sensor.get_depth()

    @property
    def marker_displacement(self):
        return self._sensor.get_marker_displacement()

    @property
    def rgb(self):
        return self._sensor.get_rgb()
