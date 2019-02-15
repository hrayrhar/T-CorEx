from __future__ import absolute_import
import os


def make_sure_path_exists(path):
    dir_name = os.path.dirname(path)
    try:
        os.makedirs(dir_name)
    except OSError:
        pass
