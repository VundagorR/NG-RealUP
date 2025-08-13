# flake8: noqa
from .archs import *
from .data import *
from .models import *
from .utils import *

from .version import __version__, __gitsha__, version_info

# Явно добавить в атрибуты модуля
globals()['__version__'] = __version__
globals()['__gitsha__'] = __gitsha__
globals()['version_info'] = version_info

# flake8: noqa
# from .archs import *
# from .data import *
# from .models import *
# from .utils import *
# from .version import *
