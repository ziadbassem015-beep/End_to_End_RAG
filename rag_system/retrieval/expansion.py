"""
retrieval/expansion.py
======================
Query expansion techniques including HyDE (Hypothetical Document Embeddings)
and Multi-Query generation using the LLM client.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from rag_system.llm.generator import LLMGenerator

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Handles query expansion and query generation using LLMGenerator.
    """

    def __init__(self, generator: Optional[LLMGenerator] = None) -> None:
        self.generator = generator

    def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical document answering the query (HyDE).
        """
        if not self.generator or self.generator.provider == "mock":
            logger.info("HyDE: Generator is unconfigured or in mock mode. Skipping.")
            return query

        system_prompt = (
            "You are an expert machine learning researcher and professor. "
            "Write a short scientific passage that directly answers the user's question. "
            "Do not include introductory text, headers, or metadata. Write only the factual explanation."
        )

        logger.info("Generating HyDE document for query: %r", query)
        try:
            # We bypass the fullgenerate_answer method because we don't want context injection here
            response = self.generator.client.chat.completions.create(
                model=self.generator.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=250,
            )
            hyde_doc = response.choices[0].message.content
            if hyde_doc:
                logger.info("Successfully generated HyDE document (%d chars)", len(hyde_doc))
                return hyde_doc
        except Exception as e:
            logger.error("Failed to generate HyDE document: %s. Falling back to original query.", e)

        return query

    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate alternative versions of the query.
        """
        queries = [query]
        if not self.generator or self.generator.provider == "mock":
            logger.info("Multi-Query: Generator is unconfigured or in mock mode. Skipping.")
            return queries

        system_prompt = (
            "You are an AI assistant tasked with generating alternative search queries. "
            f"Generate exactly {num_queries} alternative search queries that are semantically identical to the user query. "
            "Format your output as a simple numbered list, one query per line. E.g.:\n"
            "1. First query variation\n"
            "2. Second query variation\n"
            "Do not write any introductory or explanatory text. Just output the list."
        )

        logger.info("Generating %d alternative queries for: %r", num_queries, query)
        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                temperature=0.4,
                max_tokens=150,
            )
            content = response.choices[0].message.content
            if content:
                # Parse list lines
                lines = content.strip().splitlines()
                parsed = []
                for line in lines:
                    line = line.strip()
                    # Remove list markers like "1. ", "2) " or "- "
                    cleaned = line
                    cleaned = cleaned.lstrip("0123456789.-*) ")
                    if cleaned:
                        parsed.append(cleaned)
                if parsed:
                    # Append original query to guarantee it is searched
                    parsed = [p for p in parsed if p.lower() != query.lower()]
                    queries.extend(parsed[:num_queries])
                    logger.info("Multi-Query: generated variations: %s", queries)
                    return queries
        except Exception as e:
            logger.error("Failed to generate alternative queries: %s. Falling back to original.", e)

        return queries
