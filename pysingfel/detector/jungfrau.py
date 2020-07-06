import re
import six

from pysingfel.util import deprecated
from .lcls import LCLSDetector

class JungfrauDetector(LCLSDetector):
    def __init__(self, *args, **kwargs):
        super(JungfrauDetector, self).__init__(*args, **kwargs)

    def _get_cbase(self):
        """Get detector calibration base object.

        Psana 1 only.
        """
        from PSCalib.CalibParsBaseJungfrauV1 import CalibParsBaseJungfrauV1
        return CalibParsBaseJungfrauV1()
