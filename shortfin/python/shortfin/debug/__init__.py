import os

# Ensure debug module is only imported in Debug mode
DEBUG_MODE = os.environ.get("SHORTFIN_CMAKE_BUILD_TYPE", "Release") == "Debug"
if not DEBUG_MODE:
    raise EnvironmentError(
        "Debug module should only be imported in Debug mode (SHORTFIN_CMAKE_BUILD_TYPE=Debug)"
    )

from . import array

__all__ = ["array"]
