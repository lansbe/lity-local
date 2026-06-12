"""Local-first, key-free web search for the agent.

Discovery via a provider chain (SearXNG → DuckDuckGo → Wikipedia), parallel
multi-source research, and readable extraction via trafilatura. Everything
degrades gracefully: with no optional dependency installed it still works
through ``urllib`` + a crude HTML strip, and with no network it returns clear
error messages instead of raising.
"""

from lity.services.web.fetch import PageFetcher, select_relevant
from lity.services.web.research import WebResearcher
from lity.services.web.search import (
    DuckDuckGoProvider,
    SearxngProvider,
    WebSearcher,
    WikipediaProvider,
)

__all__ = [
    "PageFetcher",
    "WebResearcher",
    "select_relevant",
    "WebSearcher",
    "SearxngProvider",
    "DuckDuckGoProvider",
    "WikipediaProvider",
]
