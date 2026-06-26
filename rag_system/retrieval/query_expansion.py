"""
retrieval/query_expansion.py
============================
Query expansion techniques including HyDE (Hypothetical Document Embeddings)
and Multi-Query generation using the LLM client.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from rag_system.llm.generator import LLMGenerator

logger = logging.getLogger(__name__)


class QueryExpansion:
    """
    Generates alternative search query variations using the LLM client.
    Caches results in memory to avoid redundant calls.
    """

    def __init__(self, generator: Optional[LLMGenerator] = None) -> None:
        self.generator = generator
        self._cache: Dict[str, List[str]] = {}

    def expand_query(self, query: str, num_variants: int = 4) -> List[str]:
        """
        Generate semantically similar query variations.
        """
        cache_key = f"{query}||{num_variants}"
        if cache_key in self._cache:
            logger.debug("QueryExpansion cache hit for query: %r", query)
            return self._cache[cache_key]

        variants = [query]
        if not self.generator or self.generator.provider == "mock":
            logger.info("QueryExpansion: LLM generator is offline or mock. Skipping expansion.")
            return variants

        system_prompt = (
            "You are an AI assistant tasked with generating alternative search queries. "
            f"Generate exactly {num_variants} alternative search queries that are semantically identical to the user query. "
            "Format your output as a simple numbered list, one query per line. E.g.:\n"
            "1. First query variation\n"
            "2. Second query variation\n"
            "Do not write any introductory or explanatory text. Just output the list."
        )

        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                temperature=0.4,
                max_tokens=200,
            )
            content = response.choices[0].message.content
            if content:
                lines = content.strip().splitlines()
                parsed = []
                for line in lines:
                    line = line.strip()
                    cleaned = line.lstrip("0123456789.-*) ")
                    if cleaned:
                        parsed.append(cleaned)
                if parsed:
                    parsed = [p for p in parsed if p.lower() != query.lower()]
                    variants.extend(parsed[:num_variants])
        except Exception as e:
            logger.error("QueryExpansion: Failed to generate query variants: %s", e)

        self._cache[cache_key] = variants
        return variants


class HyDERetriever:
    """
    Generates a hypothetical document (HyDE) to answer the query.
    Embeds the hypothetical document to improve dense vector search alignment.
    """

    def __init__(self, generator: Optional[LLMGenerator] = None) -> None:
        self.generator = generator
        self._cache: Dict[str, str] = {}

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document answering the query.
        """
        if query in self._cache:
            logger.debug("HyDERetriever cache hit for query: %r", query)
            return self._cache[query]

        if not self.generator or self.generator.provider == "mock":
            logger.info("HyDERetriever: LLM generator is offline or mock. Skipping HyDE.")
            return query

        system_prompt = (
            "You are an expert machine learning researcher and professor. "
            "Write a short scientific passage that directly answers the user's question. "
            "Do not include introductory text, headers, or metadata. Write only the factual explanation."
        )

        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            hyde_doc = response.choices[0].message.content
            if hyde_doc:
                hyde_doc = hyde_doc.strip()
                self._cache[query] = hyde_doc
                return hyde_doc
        except Exception as e:
            logger.error("HyDERetriever: Failed to generate hypothetical document: %s", e)

        return query
