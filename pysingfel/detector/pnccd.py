import re
import six

from pysingfel.util import deprecated
from .lcls import LCLSDetector

class PnccdDetector(LCLSDetector):
    def __init__(self, *args, **kwargs):
        super(PnccdDetector, self).__init__(*args, **kwargs)

    def _get_cbase(self):
        """Get detector calibration base object.

        Psana 1 only.
        """
        from PSCalib.CalibParsBasePnccdV1 import CalibParsBasePnccdV1
        return CalibParsBasePnccdV1()

    def _get_det_id(self, source):
        """Get detector ID form source.

        Example: PNCCD::CalibV1 -> Camp.0:pnCCD.1 => pnccd_0001.
        Psana 2 only.
        """
        match = re.match(r"Camp\.0:pnCCD\.(\d)", source)
        number = str.zfill(match.groups()[0], 4)
        return "pnccd_" + number
