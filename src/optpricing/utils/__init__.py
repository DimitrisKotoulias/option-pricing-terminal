"""Utility helpers (visualization; requires the ``viz`` extra).

Like :mod:`optpricing.data`, this subpackage deliberately does **not**
re-export its members at the package level. Import them explicitly, e.g.::

    from optpricing.utils.visualization import ...

This keeps ``import optpricing`` from eagerly pulling in the optional ``viz``
dependency (``matplotlib``), so the core pricing library has no heavy imports.
"""
