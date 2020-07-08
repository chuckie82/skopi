import re
import six
from pysingfel.util import deprecated
from .lcls import LCLSDetector

class AutoRangingDetector(LCLSDetector):
    def __init__(self, cameraConfig, *args, **kwargs):
        super(AutoRangingDetector, self).__init__(*args, **kwargs)
        self.cameraConfig = cameraConfig
