from pathlib import Path
import datetime


class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

    @property
    def root(self):
        return self._root

    @property
    def output(self):
        result = (self.root / 'output')
        result.mkdir(parents=True, exist_ok = True)
        return result

project = CustomizedPath() 