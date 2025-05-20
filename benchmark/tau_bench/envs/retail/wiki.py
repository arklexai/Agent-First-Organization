# Copyright Sierra

import os
from typing import Final

FOLDER_PATH: Final[str] = os.path.dirname(__file__)

with open(os.path.join(FOLDER_PATH, "wiki.md"), "r") as f:
    WIKI: Final[str] = f.read()
