import re
import six

from pysingfel.util import deprecated
from .lcls import LCLSDetector

class Epix10kDetector(LCLSDetector):
    def __init__(self, *args, **kwargs):
        super(Epix10kDetector, self).__init__(*args, **kwargs)

    def _get_cbase(self):
        """Get detector calibration base object.

        Psana 1 only.
        """
        from PSCalib.CalibParsBaseEpix10kaV1 import CalibParsBaseEpix10kaV1
        return CalibParsBaseEpix10kaV1()
