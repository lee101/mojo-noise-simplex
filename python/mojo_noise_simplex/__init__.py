"""noise-simplex, ported to Mojo.

    import mojo_noise_simplex as mns

    mns.snoise2(0.3, 0.7)                    # float, like noise.snoise2
    mns.snoise2(xs, ys)                      # array field, broadcasting
    mns.snoise2(xs, ys, repeatx=8, repeaty=8)  # tileable field

Upstream is Casey Duncan's `noise` package; see the README for the exact
subset covered and the divergences.
"""

from .simplex import FLT_MAX, snoise2, snoise3, snoise4

__all__ = ["snoise2", "snoise3", "snoise4", "FLT_MAX", "__version__"]
__version__ = "0.1.0"
