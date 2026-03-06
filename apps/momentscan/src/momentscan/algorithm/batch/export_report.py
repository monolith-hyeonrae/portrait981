"""Interactive highlight analysis HTML report — delegated to momentscan-report.

This module re-exports from momentscan_report.highlight for backward compatibility.
"""

from momentscan_report.highlight import *  # noqa: F401, F403
from momentscan_report.highlight import (  # noqa: F401
    export_highlight_report,
    _build_chart_data,
    _build_summary_html,
    _build_window_detail_html,
    _build_config_table_html,
    _build_field_reference_html,
    _select_thumbnail_indices,
    _JS_MAIN,
)
