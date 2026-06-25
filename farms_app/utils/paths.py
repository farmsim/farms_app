""" Paths are structured in the repository """

import functools
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def add_project_root(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ROOT_PATH = get_project_root()
        return ROOT_PATH.joinpath(func(*args, **kwargs))
    return wrapper
