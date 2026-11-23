"""Apple Accelerate's LAPACK for the host factorisation of `jax_ls.nd_solver`.

``factor_on="accelerate"`` factors the fronts of the ``"block_inverse"``
preconditioner on the host CPU, like ``factor_on="cpu"``, but with Accelerate
instead of whatever LAPACK SciPy was built against.  Each level of the
dissection tree is one call into a small C shim (``_accelerate_fronts.c``,
shipped with the package), which factors its fronts in parallel.

It pays off when NumPy and SciPy are not already linked against Accelerate --
a default Anaconda install uses OpenBLAS; check with ``numpy.show_config()``
and ``scipy.show_config()``.  The pip wheels of NumPy 2.0+ and recent SciPy on
Apple Silicon with macOS 14+ already use Accelerate, so ``factor_on="cpu"``
does too, and the only gain left is the parallel loop over fronts.

macOS 13.3 or later only.  The shim is compiled on first use with the system
compiler and cached, so it needs the Xcode Command Line Tools
(``xcode-select --install``).  Set ``JAX_LS_CACHE_DIR`` to choose where the
compiled library goes.
"""

import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

MINIMUM_MACOS = (13, 3)

_SOURCE = Path(__file__).with_name("_accelerate_fronts.c")
_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None


def unavailable_reason() -> str | None:
  """Why Accelerate cannot be used here, without compiling anything.

  Returns:
    None when the platform supports it; otherwise a short explanation.
  """
  if sys.platform != "darwin":
    return f"Accelerate is macOS only; this is {sys.platform}"
  release = platform.mac_ver()[0]
  version = tuple(int(part) for part in release.split(".") if part.isdigit())
  if version < MINIMUM_MACOS:
    return (
      f"Accelerate's new LAPACK interface needs macOS 13.3+, found {release}"
    )
  return None


def available() -> bool:
  """Whether Accelerate can be used here.

  Compiles the shim if needed, so a missing compiler also reports False.

  Returns:
    True when `load` succeeds.
  """
  try:
    load()
  except RuntimeError:
    return False
  return True


def _cache_dir() -> Path:
  """Where the compiled shim is kept.

  Returns:
    ``$JAX_LS_CACHE_DIR``, else ``~/Library/Caches/jax_ls``.
  """
  configured = os.environ.get("JAX_LS_CACHE_DIR")
  if configured:
    return Path(configured)
  return Path.home() / "Library" / "Caches" / "jax_ls"


def _compile(target: Path) -> None:
  """Compile the shim into `target`, atomically.

  Args:
    target: Path of the shared library to produce.

  Raises:
    RuntimeError: If the compiler is missing or fails.
  """
  target.parent.mkdir(parents=True, exist_ok=True)
  handle, scratch = tempfile.mkstemp(suffix=".dylib", dir=target.parent)
  os.close(handle)
  command = [
    "cc",
    "-O2",
    "-shared",
    "-fPIC",
    "-mmacosx-version-min=13.3",
    "-framework",
    "Accelerate",
    str(_SOURCE),
    "-o",
    scratch,
  ]
  try:
    result = subprocess.run(command, capture_output=True, text=True)
  except FileNotFoundError as error:
    os.unlink(scratch)
    raise RuntimeError(
      "no C compiler found to build the Accelerate shim; install the Xcode "
      "Command Line Tools with `xcode-select --install`"
    ) from error
  if result.returncode != 0:
    os.unlink(scratch)
    raise RuntimeError(
      f"compiling the Accelerate shim failed:\n{result.stderr.strip()}"
    )
  os.replace(scratch, target)


def load() -> ctypes.CDLL:
  """The compiled shim, built and cached on first use.

  Returns:
    The loaded library, with its argument types declared.

  Raises:
    RuntimeError: Off macOS, before macOS 13.3, or when compilation fails.
  """
  global _LIBRARY
  with _LOCK:
    if _LIBRARY is not None:
      return _LIBRARY
    reason = unavailable_reason()
    if reason is not None:
      raise RuntimeError(reason)
    source = _SOURCE.read_bytes()
    digest = hashlib.sha256(source + platform.machine().encode()).hexdigest()
    target = _cache_dir() / f"accelerate_fronts-{digest[:16]}.dylib"
    if not target.exists():
      _compile(target)
    library = ctypes.CDLL(str(target))
    library.acc_factor_fronts.restype = ctypes.c_int
    library.acc_factor_fronts.argtypes = [ctypes.c_int] * 3 + [
      ctypes.c_void_p
    ] * 9
    _LIBRARY = library
    return library


def factor_fronts(
  f_ss: np.ndarray, f_sb: np.ndarray, f_bs: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Factor a level of fronts: the stored blocks and the pivoting diagnostics.

  Args:
    f_ss: Separator blocks, ``(k, s, s)``.
    f_sb: Separator-to-boundary blocks, ``(k, s, b)``.
    f_bs: Boundary-to-separator blocks, ``(k, b, s)``.
    valid: Which separator points are real rather than padding, ``(k, s)``.

  Returns:
    ``(inverse, upper, lower, growth, pivot_ratio)``: ``F_SS^{-1}``,
    ``F_SS^{-1} F_SB`` and ``F_BS F_SS^{-1}``, complex128, and per-front
    element growth and smallest-to-largest pivot ratio over the valid points.

  Raises:
    np.linalg.LinAlgError: If a front's ``F_SS`` is exactly singular.
    MemoryError: If the shim could not allocate its workspace.
  """
  library = load()
  k, s, b = f_sb.shape
  f_ss = np.ascontiguousarray(f_ss, dtype=np.complex128)
  f_sb = np.ascontiguousarray(f_sb, dtype=np.complex128)
  f_bs = np.ascontiguousarray(f_bs, dtype=np.complex128)
  valid = np.ascontiguousarray(valid, dtype=np.uint8)
  inverse = np.empty((k, s, s), dtype=np.complex128)
  upper = np.empty((k, s, b), dtype=np.complex128)
  lower = np.empty((k, b, s), dtype=np.complex128)
  growth = np.empty(k)
  pivot_ratio = np.empty(k)
  status = library.acc_factor_fronts(
    k,
    s,
    b,
    *(
      array.ctypes.data
      for array in (
        f_ss,
        f_sb,
        f_bs,
        valid,
        inverse,
        upper,
        lower,
        growth,
        pivot_ratio,
      )
    ),
  )
  if status == -1:
    raise MemoryError("the Accelerate shim could not allocate its workspace")
  if status > 0:
    raise np.linalg.LinAlgError(
      f"a front's diagonal block is exactly singular (front {status - 1})"
    )
  return inverse, upper, lower, growth, pivot_ratio
