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

    def _get_det_id(self, source):
        """Get detector ID form source.

        Example: CsPad::CalibV1 -> CxiDs2.0:Cspad.0 => cspad_0002.
        Psana 2 only.
        """
        match = re.match(r"CxiDs(\d)\.0:Cspad\.0", source)
        number = str.zfill(match.groups()[0], 4)
        return "cspad_" + number
