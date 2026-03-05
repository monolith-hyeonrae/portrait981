"""Portrait generation orchestrator."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from reportrait.comfy_client import ComfyClient
from reportrait.types import GenerationConfig, GenerationRequest, GenerationResult
from reportrait.workflow import inject_prompt, inject_references, load_template

logger = logging.getLogger(__name__)


class PortraitGenerator:
    """Orchestrates portrait generation via ComfyUI.

    load template -> inject refs -> inject prompt -> queue -> wait -> download.
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.client = ComfyClient(self.config.comfy_url, api_key=self.config.api_key)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate portrait images from a request.

        Args:
            request: GenerationRequest with person_id, ref_paths, etc.

        Returns:
            GenerationResult with output paths or error.
        """
        start = time.monotonic()

        try:
            # Load workflow template
            workflow = load_template(
                request.workflow_template,
                templates_dir=self.config.templates_dir,
            )

            # Inject reference images and prompt
            workflow = inject_references(workflow, request.ref_paths, node_ids=request.node_ids)
            if request.style_prompt:
                workflow = inject_prompt(workflow, request.style_prompt)

            # Queue and wait
            prompt_id = self.client.queue_prompt(workflow)
            logger.info("Queued prompt %s for person %d", prompt_id, request.person_id)

            history = self.client.wait_for_completion(
                prompt_id,
                timeout=self.config.timeout_sec,
                poll_interval=self.config.poll_interval_sec,
            )

            # Download results
            output_dir = self.config.output_dir or Path("output") / f"person_{request.person_id}"
            paths = self.client.download_images(history, output_dir)

            elapsed = time.monotonic() - start
            return GenerationResult(
                success=True,
                output_paths=[str(p) for p in paths],
                workflow_used=workflow,
                elapsed_sec=elapsed,
            )

        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error("Generation failed for person %d: %s", request.person_id, e)
            return GenerationResult(
                success=False,
                error=str(e),
                elapsed_sec=elapsed,
            )

    def generate_from_bank(
        self,
        bank_path: Path,
        *,
        query: Optional[dict] = None,
        style_prompt: str = "",
        workflow_template: Optional[str] = None,
    ) -> GenerationResult:
        """Generate from a saved MemoryBank.

        Loads the bank, selects reference images, then generates.

        Args:
            bank_path: Path to memory_bank.json.
            query: Optional target_buckets for RefQuery, e.g. {"yaw": "[-5,5]"}.
            style_prompt: Optional style prompt.
            workflow_template: Override workflow template name.

        Returns:
            GenerationResult.
        """
        from momentbank import load_bank, RefQuery

        bank = load_bank(bank_path)
        ref_query = RefQuery(target_buckets=query or {})
        selection = bank.select_refs(ref_query)

        if not selection.paths_for_comfy:
            return GenerationResult(
                success=False,
                error="No reference images available in bank",
            )

        request = GenerationRequest(
            person_id=bank.person_id,
            ref_paths=selection.paths_for_comfy,
            workflow_template=workflow_template or self.config.workflow_template,
            style_prompt=style_prompt,
        )

        return self.generate(request)

    def generate_from_lookup(
        self,
        member_id: str,
        *,
        pose: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 3,
        style_prompt: str = "",
        workflow_template: Optional[str] = None,
    ) -> GenerationResult:
        """Generate from lookup_frames() query.

        Uses momentbank.lookup_frames to find reference images by pose/category,
        then generates via ComfyUI.

        Args:
            member_id: Member identifier for frame lookup.
            pose: Filter by pose_name (e.g. "left30", "frontal"). None = all.
            category: Filter by category (e.g. "warm_smile"). None = all.
            top_k: Max reference images to use.
            style_prompt: Optional style prompt.
            workflow_template: Override workflow template name.

        Returns:
            GenerationResult.
        """
        from momentbank.ingest import lookup_frames

        frames = lookup_frames(member_id, pose=pose, category=category, top_k=top_k)

        if not frames:
            return GenerationResult(
                success=False,
                error=f"No frames found for member '{member_id}'"
                + (f" pose={pose}" if pose else "")
                + (f" category={category}" if category else ""),
            )

        ref_paths = [f["path"] for f in frames]

        request = GenerationRequest(
            person_id=0,
            ref_paths=ref_paths,
            workflow_template=workflow_template or self.config.workflow_template,
            style_prompt=style_prompt,
        )

        return self.generate(request)
