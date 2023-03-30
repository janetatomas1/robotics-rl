import csv
import json


class Logger:
    def __init__(self, file_mode):
        self._file_mode = file_mode
        self._path = None
        self._file = None

    def save_history(self, **kwargs):
        pass

    def close(self):
        self._file.close()

    def open(self, path):
        self._path = path
        self._file = open(self._path, self._file_mode)


class JsonLogger(Logger):
    def __init__(self):
        super().__init__("w")

    def save_history(self, **kwargs):
        self._file.write(f'{json.dumps(kwargs)}\n')
        self._file.flush()


class CSVLogger(Logger):
    def __init__(self):
        super().__init__("w")
        self._writer = None

    def open(self, path):
        super().open(path)
        self._writer = csv.writer(self._file)

    def save_history(self, **kwargs):
        row = list()
        for _, x in list(kwargs.items()):
            if type(x) == list:
                row += x
            else:
                row.append(x)

        self._writer.writerow(row)
        self._file.flush()
