"""VQLS"""

from importlib_metadata import version as metadata_version, PackageNotFoundError

from .solver.vqls import VQLS
from .solver.log import VQLSLog
from .solver.hybrid_qst_vqls import Hybrid_QST_VQLS
from .solver.qst_vqls import QST_VQLS


try:
    __version__ = metadata_version("vqls_prototype")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass

__all__ = ["VQLS", "VQLSLog", "Hybrid_QST_VQLS", "QST_VQLS"]
