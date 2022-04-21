from .base import DetectorBase
from .plain import PlainDetector
from .user_defined import UserDefinedDetector
from .simple_square import SimpleSquareDetector

global psana_version
try:
    from PSCalib.GeometryAccess import GeometryAccess
    psana_version=1
except Exception:
    try:
        from psana.pscalib.geometry.GeometryAccess import GeometryAccess
        psana_version=2
    except:
        # psana unavailable; skip all AutorangingDetector tests
        psana_version=0

if psana_version > 0:
    from .lcls import LCLSDetector
    from .pnccd import PnccdDetector
    from .cspad import CsPadDetector
    from .epix10k import Epix10kDetector
    from .jungfrau import JungfrauDetector
