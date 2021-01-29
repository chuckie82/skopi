import re
import six

from pysingfel.util import deprecated
from .lcls import LCLSDetector

class CsPadDetector(LCLSDetector):
    def __init__(self, *args, **kwargs):
        super(CsPadDetector, self).__init__(*args, **kwargs)

    def _get_cbase(self):
        """Get detector calibration base object.

        Psana 1 only.
        """
        from PSCalib.CalibParsBaseCSPadV1 import CalibParsBaseCSPadV1
        return CalibParsBaseCSPadV1()

    def _get_det_id(self, group):
        """Get detector ID form group.

        Example: CsPad::CalibV1 -> cspad_0001.
        Psana 2 only.
        """
        match = re.match(r"CsPad::CalibV(\d)", group)
        number = str.zfill(match.groups()[0], 4)
        return "cspad_" + number

