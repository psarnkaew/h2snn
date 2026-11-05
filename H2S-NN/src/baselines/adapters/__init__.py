"""One module per baseline, each exposing an ``ADAPTER`` instance.

Adapters are imported lazily by :mod:`src.baselines.registry` so that a missing vendor
clone or an unavailable optional dependency only breaks the cells that need it, rather
than every import of the package.
"""
