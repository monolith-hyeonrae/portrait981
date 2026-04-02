"""Standard ROI specifications for face analysis pipelines.

Defines the official ROI hierarchy. Analyzers reference these by name
via Module.roi_requires instead of computing crops independently.

Hierarchy: Global ⊃ Body ⊃ Portrait ⊃ Head ⊃ Face
"""

from visualbase.core.roi import ROISpec

# Tight face bbox — AU, expression analysis
FACE = ROISpec("face", expand=1.0, size=(224, 224), aspect_ratio=1.0)

# Head region — head pose estimation (slightly expanded)
HEAD = ROISpec("head", expand=1.5, size=(224, 224), aspect_ratio=1.0)

# Portrait crop — face + neck + shoulders, for segmentation/quality/lighting
PORTRAIT = ROISpec("portrait", expand=1.3, size=(512, 512), aspect_ratio=1.0)

# Registry for lookup by name
ROI_REGISTRY: dict[str, ROISpec] = {
    "face": FACE,
    "head": HEAD,
    "portrait": PORTRAIT,
}


def get_roi_spec(name: str) -> ROISpec:
    """Lookup ROI spec by name. Raises KeyError if not found."""
    return ROI_REGISTRY[name]
