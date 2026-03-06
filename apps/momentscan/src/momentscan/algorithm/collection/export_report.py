"""Interactive collection analysis HTML report — delegated to momentscan-report.

This module re-exports from momentscan_report.collection for backward compatibility.
"""

from momentscan_report.collection import *  # noqa: F401, F403
from momentscan_report.collection import (  # noqa: F401
    export_collection_report,
    _build_persons_data,
    _build_summary_html,
    _build_gallery_html,
    _build_config_table_html,
    _collect_all_selected,
    _JS_MAIN,
)
