"""face.lighting — Portrait lighting analysis.

BiSeNet face mask를 이용한 얼굴 조명 패턴 분석.
Rembrandt, split, loop, flat 등의 조명 패턴을 감지.

Usage:
    from momentscan.face_lighting import FaceLightingAnalyzer
"""

from momentscan.face_lighting.analyzer import FaceLightingAnalyzer

__all__ = ["FaceLightingAnalyzer"]
