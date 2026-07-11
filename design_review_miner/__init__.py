"""design-review-miner — AI-assisted retrieval of past engineering lessons.

Developed by Dr. Fitsum Taye Feyissa.
"""

from .index import KnowledgeIndex, Match, keyword_profile

__version__ = "1.0.0"
__all__ = ["KnowledgeIndex", "Match", "keyword_profile"]
