"""VQLS"""

from importlib_metadata import version as metadata_version, PackageNotFoundError

from .solver.vqls import VQLS
from .solver.log import VQLSLog


try:
    __version__ = metadata_version("vqls_prototype")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass

__all__ = ["VQLS", "VQLSLog"]
