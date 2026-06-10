from importlib.metadata import PackageNotFoundError, version

from .model import SPACE

try:
    __version__ = version('space-graph')
except PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = ['SPACE', '__version__']
