"""
evaluation/validator.py
========================
Evaluation dataset validator.
Detects invalid chunk references, semantic coverage issues, and irrelevant chunks
(TOC pages, copyright pages, metadata/blank pages) using rule-based heuristics.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class IrrelevantChunkDetails:
    """Details of an irrelevant chunk detected in relevant_chunk_ids."""
    chunk_id: str
    reason: str
    content_preview: str
    page: int


@dataclass
class QueryValidation:
    """Validation result for a single query."""
    query_id: str
    query: str
    annotated_relevant_ids: List[str]
    missing_chunk_ids: List[str]
    valid_chunk_ids: List[str]
    avg_semantic_similarity: float
    suggested_relevant_ids: List[str]
    irrelevant_chunks: List[IrrelevantChunkDetails]
    is_valid: bool


@dataclass
class ValidationReport:
    """Full validation report for an eval dataset."""
    total_queries: int
    valid_queries: int
    queries_with_missing_ids: int
    queries_with_low_similarity: int
    queries_with_irrelevant_chunks: int
    missing_chunk_ids: List[str]
    coverage_score: float
    low_similarity_threshold: float
    per_query: List[QueryValidation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_queries": self.total_queries,
                "valid_queries": self.valid_queries,
                "queries_with_missing_ids": self.queries_with_missing_ids,
                "queries_with_low_similarity": self.queries_with_low_similarity,
                "queries_with_irrelevant_chunks": self.queries_with_irrelevant_chunks,
                "coverage_score": round(self.coverage_score, 4),
                "low_similarity_threshold": self.low_similarity_threshold,
            },
            "missing_chunk_ids": sorted(set(self.missing_chunk_ids)),
            "per_query": [
                {
                    "query_id": q.query_id,
                    "query": q.query,
                    "annotated_relevant_ids": q.annotated_relevant_ids,
                    "missing_chunk_ids": q.missing_chunk_ids,
                    "valid_chunk_ids": q.valid_chunk_ids,
                    "avg_semantic_similarity": round(q.avg_semantic_similarity, 4),
                    "suggested_relevant_ids": q.suggested_relevant_ids,
                    "irrelevant_chunks": [asdict(ic) for ic in q.irrelevant_chunks],
                    "is_valid": q.is_valid,
                }
                for q in self.per_query
            ],
        }


class EvalDatasetValidator:
    """
    Validator to check the health and correctness of evaluation datasets.
    """

    def __init__(
        self,
        embeddings_path: str | Path,
        model_name: str = "BAAI/bge-small-en-v1.5",
        low_similarity_threshold: float = 0.35,
        suggestion_top_k: int = 5,
    ) -> None:
        self._threshold = low_similarity_threshold
        self._suggestion_top_k = suggestion_top_k

        logger.info("Loading embeddings from %s for validation", embeddings_path)
        path = Path(embeddings_path)
        
        # Fallback logic if the requested file doesn't exist
        if not path.exists():
            faiss_metadata = Path("data/indices/faiss/andrew-ng-machine-learning-yearning/metadata.json")
            if faiss_metadata.exists():
                logger.info("Embeddings file not found. Falling back to FAISS metadata: %s", faiss_metadata)
                path = faiss_metadata
            else:
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path} and fallback FAISS metadata not found.")

        # If it's a directory, look for metadata.json inside it
        if path.is_dir():
            path = path / "metadata.json"

        if path.name == "metadata.json":
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self._embedded_chunks = list(meta.get("metadata", {}).values())
            logger.info("Loaded %d embedded chunks from FAISS metadata.", len(self._embedded_chunks))
        else:
            from rag_system.embeddings.embedder import Embedder
            self._embedded_chunks = Embedder.load(path)

        # Build lookup maps
        self._id_to_chunk = {c["chunk_id"]: c for c in self._embedded_chunks}
        self._all_ids = set(self._id_to_chunk.keys())

        # Build matrix
        vecs = np.array([c["embedding"] for c in self._embedded_chunks], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self._matrix = vecs / norms
        self._ordered_ids = [c["chunk_id"] for c in self._embedded_chunks]

        logger.info("Loading query encoder for validator: %s", model_name)
        self._encoder = SentenceTransformer(model_name)
        logger.info("Validator ready with %d chunks indexed", len(self._embedded_chunks))

    def validate(self, eval_dataset_path: str | Path) -> ValidationReport:
        """
        Validate an evaluation dataset.
        """
        path = Path(eval_dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {path}")
            
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Validating %d queries from %s", len(data), path)

        per_query: List[QueryValidation] = []
        all_missing: List[str] = []

        for idx, item in enumerate(data):
            qid = item.get("query_id", f"q_{idx:03d}")
            query = item["query"]
            annotated_ids = [str(cid) for cid in item.get("relevant_chunk_ids", [])]

            # 1. Missing/Invalid Chunk IDs
            missing = [cid for cid in annotated_ids if cid not in self._all_ids]
            valid = [cid for cid in annotated_ids if cid in self._all_ids]
            all_missing.extend(missing)

            # 2. Semantic Coverage Check (Cosine Sim query ↔ valid annotated chunks)
            avg_sim = self._compute_avg_similarity(query, valid)

            # 3. Detect Irrelevant Chunks (TOC, Copyright, Metadata)
            irrelevant_chunks = []
            for cid in valid:
                chunk = self._id_to_chunk[cid]
                content = chunk["content"]
                
                # Check top-level or metadata-nested page number
                page = int(chunk.get("page", chunk.get("metadata", {}).get("page", 1)))
                
                is_irr, reason = self.is_irrelevant_chunk(content, page)
                if is_irr:
                    irrelevant_chunks.append(
                        IrrelevantChunkDetails(
                            chunk_id=cid,
                            reason=reason,
                            content_preview=content[:100].replace("\n", " ") + "...",
                            page=page,
                        )
                    )

            # 4. Suggest valid replacement chunks
            suggested = self._suggest_relevant(query, top_k=self._suggestion_top_k)

            # Flag as invalid if there are missing IDs, low similarity, or irrelevant chunks referenced
            is_valid = (
                len(missing) == 0
                and avg_sim >= self._threshold
                and len(irrelevant_chunks) == 0
            )

            per_query.append(
                QueryValidation(
                    query_id=qid,
                    query=query,
                    annotated_relevant_ids=annotated_ids,
                    missing_chunk_ids=missing,
                    valid_chunk_ids=valid,
                    avg_semantic_similarity=avg_sim,
                    suggested_relevant_ids=suggested,
                    irrelevant_chunks=irrelevant_chunks,
                    is_valid=is_valid,
                )
            )

        valid_count = sum(1 for q in per_query if q.is_valid)
        missing_q = sum(1 for q in per_query if q.missing_chunk_ids)
        low_sim_q = sum(1 for q in per_query if q.avg_semantic_similarity < self._threshold)
        irr_q = sum(1 for q in per_query if q.irrelevant_chunks)

        report = ValidationReport(
            total_queries=len(per_query),
            valid_queries=valid_count,
            queries_with_missing_ids=missing_q,
            queries_with_low_similarity=low_sim_q,
            queries_with_irrelevant_chunks=irr_q,
            missing_chunk_ids=all_missing,
            coverage_score=valid_count / len(per_query) if per_query else 0.0,
            low_similarity_threshold=self._threshold,
            per_query=per_query,
        )

        return report

    def generate_corrected_dataset(
        self,
        eval_dataset_path: str | Path,
        output_path: str | Path,
        top_k: int = 5,
    ) -> Path:
        """
        Generate a corrected eval dataset based on top-k semantic suggestions.
        """
        path = Path(eval_dataset_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        corrected = []
        for idx, item in enumerate(data):
            qid = item.get("query_id", f"q_{idx:03d}")
            query = item["query"]

            # Suggest chunks, but filter out chunks detected as irrelevant (TOC/copyright)
            suggested = []
            q_vec = self._embed_query(query)
            scores = self._matrix @ q_vec
            top_indices = np.argsort(scores)[::-1]

            for index in top_indices:
                cid = self._ordered_ids[index]
                chunk = self._id_to_chunk[cid]
                content = chunk["content"]
                page = int(chunk.get("page", chunk.get("metadata", {}).get("page", 1)))
                
                is_irr, _ = self.is_irrelevant_chunk(content, page)
                if not is_irr:
                    suggested.append(cid)
                    if len(suggested) >= top_k:
                        break

            corrected.append({
                "query_id": qid,
                "query": query,
                "relevant_chunk_ids": suggested,
                "original_relevant_chunk_ids": item.get("relevant_chunk_ids", []),
            })

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(corrected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved corrected evaluation dataset to: %s", out)
        return out

    def save_report(
        self,
        report: ValidationReport,
        output_path: str | Path,
    ) -> Path:
        """Save report to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved validation report to: %s", out)
        return out

    @staticmethod
    def is_irrelevant_chunk(content: str, page: int) -> Tuple[bool, str]:
        """
        Rule-based heuristic to check if the chunk is TOC, copyright, or blank/metadata front matter.
        """
        content_lower = content.lower()

        # 1. Check for copyright indicator
        if page > 0 and page <= 6:
            copyright_indicators = [
                "copyright", "all rights reserved", "isbn", "published by", "deeplearning.ai"
            ]
            for ind in copyright_indicators:
                if ind in content_lower:
                    return True, f"Early page ({page}) containing copyright text: '{ind}'"

        # 2. Table of Contents Keywords
        if page > 0 and page <= 6:
            if "contents" in content_lower or "table of contents" in content_lower:
                return True, f"Early page ({page}) containing TOC keywords"

        # 3. Text layout check (high density of dotted lines or page number endings)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines:
            dotted_lines = sum(
                1 for line in lines
                if "..." in line or "···" in line or re.search(r'\s+\d+$', line)
            )
            if dotted_lines / len(lines) > 0.4:
                return True, f"Heuristic: Dotted line/page number layout density ({dotted_lines / len(lines):.2f}) resembling TOC"

        # 4. Tiny chunks
        if len(content.strip()) < 40:
            return True, "Chunk too short (less than 40 chars) - likely header, footer, or metadata artifact"

        return False, ""

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self._encoder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vec.astype(np.float32)

    def _compute_avg_similarity(self, query: str, chunk_ids: List[str]) -> float:
        if not chunk_ids:
            return 0.0
        q_vec = self._embed_query(query)
        sims = []
        for cid in chunk_ids:
            if cid in self._id_to_chunk:
                c_vec = np.array(self._id_to_chunk[cid]["embedding"], dtype=np.float32)
                c_norm = c_vec / (np.linalg.norm(c_vec) + 1e-10)
                sims.append(float(np.dot(q_vec, c_norm)))
        return float(np.mean(sims)) if sims else 0.0

    def _suggest_relevant(self, query: str, top_k: int) -> List[str]:
        q_vec = self._embed_query(query)
        scores = self._matrix @ q_vec
        k = min(top_k, len(self._ordered_ids))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self._ordered_ids[int(i)] for i in top_idx]
