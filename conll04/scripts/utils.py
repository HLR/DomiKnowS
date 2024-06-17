import sys
import pathlib
import inspect


def add_example_path():
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

def current_path():
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    return str(pathlib.Path(module.__file__).parent)
