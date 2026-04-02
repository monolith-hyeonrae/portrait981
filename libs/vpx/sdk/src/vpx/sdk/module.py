"""Module base class for vpx analyzers.

Extends visualpath's Module with ROI requirements and analyze() alias.
"""

from typing import List, Optional

from visualpath.core.module import Module as VisualPathModule


class Module(VisualPathModule):
    """Base class for vpx analyzer modules.

    Adds:
    - roi_requires: declare which ROI crops this analyzer needs
    - analyze(): backwards-compatible alias for process()
    """

    # ROI requirements: list of ROI names this analyzer needs (e.g. ["face", "portrait"])
    # ROIs are created by upstream modules (e.g. face.detect) and routed via deps.
    roi_requires: List[str] = []

    def analyze(self, frame: "Frame", deps=None) -> Optional["Observation"]:
        """Backwards-compatible alias for process()."""
        return self.process(frame, deps)


# Backwards compatibility alias
BaseAnalyzer = Module
