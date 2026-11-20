# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Forwarding shim. Both FlagAudio and FlagGems register their runtime
# backend directory on sys.path and share the top-level package name
# "_mthreads"; FlagAudio's package takes precedence but only vendors a
# subset of modules, so FlagGems' resolution of
# "_mthreads.heuristics_config_utils" lands here.
#
# The real implementation is loaded directly from its file instead of
# going through the flag_gems package chain: a normal package import here
# would trigger full flag_gems initialization while this shim is itself
# being imported during that same initialization (circular import), and
# FlagGems would observe a partially initialized module without
# HEURISTICS_CONFIGS. The real module only depends on torch/triton, so
# loading it from file is side-effect free and keeps a single source of
# truth.
import importlib.util
import sys
from pathlib import Path

_REAL_MODULE_NAME = "flag_gems.runtime.backend._mthreads.heuristics_config_utils"

_pkg_spec = importlib.util.find_spec("flag_gems")
if _pkg_spec is None or not _pkg_spec.submodule_search_locations:
    raise ImportError(
        "flag_gems is required to resolve the mthreads heuristics config shim"
    )

_real_path = (
    Path(_pkg_spec.submodule_search_locations[0])
    / "runtime"
    / "backend"
    / "_mthreads"
    / "heuristics_config_utils.py"
)
_real_spec = importlib.util.spec_from_file_location(_REAL_MODULE_NAME, _real_path)
_real_module = importlib.util.module_from_spec(_real_spec)
_real_spec.loader.exec_module(_real_module)
# Reuse the same module object for later fully-qualified package imports.
sys.modules.setdefault(_REAL_MODULE_NAME, _real_module)

globals().update(
    {name: value for name, value in vars(_real_module).items() if not name.startswith("_")}
)
