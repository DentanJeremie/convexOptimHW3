from datetime import datetime
import os
from pathlib import Path
import typing as t


class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent
        self._log_initialized = False
        self._sum_log_initialized = False

    def remove_prefix(input_string: str, prefix: str):
        """Needed for Python<3.9"""
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string

    def as_relative(self, path: t.Union[str, Path]):
        if type(path) == str:
            path = Path(path)
        return Path(CustomizedPath.remove_prefix(path.as_posix(), self.root.as_posix()))

    @property
    def root(self):
        return self._root

    @property
    def output(self):
        result = self.root / 'output'
        result.mkdir(parents=True, exist_ok = True)
        return result

    @property
    def mu_figure(self):
        return self.output / 'mu_performances.png'

    @property
    def log(self):
        result = self.output / 'main.log'

        # Checking if exists
        if not os.path.isfile(result):
            with result.open('w') as f:
                pass

        # Header for new log
        if not self._log_initialized:
            with result.open('a') as f:
                f.write(f'\nNEW LOG AT {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
            self._log_initialized = True

        return result

    @property
    def sum_log(self):
        result = self.output / 'summary.log'

        # Checking if exists
        if not os.path.isfile(result):
            with result.open('w') as f:
                pass

        # Header for new log
        if not self._sum_log_initialized:
            with result.open('a') as f:
                f.write(f'\nNEW LOG AT {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
            self._sum_log_initialized = True

        return result

project = CustomizedPath()