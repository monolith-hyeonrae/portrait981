"""Module base class for vpx analyzers.

Extends visualpath's Module with a backwards-compatible analyze() alias.
"""

from typing import Optional

from visualpath.core.module import Module as VisualPathModule


class Module(VisualPathModule):
    """Module with backwards-compatible analyze() alias.

    This class extends visualpath's Module to provide the legacy `analyze()`
    method as an alias for `process()`.
    """

    def analyze(self, frame: "Frame", deps=None) -> Optional["Observation"]:
        """Backwards-compatible alias for process().

        Args:
            frame: Input frame to analyze.
            deps: Optional dependency observations.

        Returns:
            Observation from processing.
        """
        return self.process(frame, deps)


# Backwards compatibility alias
BaseAnalyzer = Module
