"""Face detection and expression analysis backend implementations.

This module re-exports backends from individual files for backward compatibility.
New code should import directly from the specific modules:

    from vpx.face_detect.backends.insightface import InsightFaceSCRFD
    from vpx.expression.backends.hsemotion import HSEmotionBackend
    from vpx.expression.backends.pyfeat import PyFeatBackend
"""

# Re-export for backward compatibility (lazy to avoid ImportError when deps missing)
__all__: list[str] = []

try:
    from vpx.face_detect.backends.insightface import InsightFaceSCRFD
    __all__.append("InsightFaceSCRFD")
except ImportError:
    pass

try:
    from vpx.expression.backends.hsemotion import HSEmotionBackend
    __all__.append("HSEmotionBackend")
except ImportError:
    pass

try:
    from vpx.expression.backends.pyfeat import PyFeatBackend
    __all__.append("PyFeatBackend")
except ImportError:
    pass
