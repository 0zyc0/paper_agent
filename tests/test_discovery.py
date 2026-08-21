from paper_agent.core.discovery import DiscoveryItem, DiscoveryService
from paper_agent.core.models import Paper


class FakePaperSource:
    def __init__(self, papers):
        self.papers = papers

    def search(self, *args, **kwargs):
        return list(self.papers)


def test_discovery_service_returns_papers_links_and_directions(monkeypatch):
    service = DiscoveryService()
    service.openalex = FakePaperSource([
        Paper(
            title="Debiasing Dynamic Recommender Systems",
            authors=["Ada Lovelace"],
            abstract="A recent paper.",
            year=2026,
            venue="SIGIR",
            source="openalex",
            source_url="https://example.org/paper",
        )
    ])
    service.arxiv = FakePaperSource([])
    service.dblp = FakePaperSource([])
    monkeypatch.setattr(
        service,
        "_rss_items",
        lambda topic: ([DiscoveryItem("技术解读", "CSDN", "https://example.org/blog", kind="article")], []),
    )

    payload = service.discover("debiasing dynamic recommender systems")

    assert payload["papers"][0]["title"] == "Debiasing Dynamic Recommender Systems"
    assert any(item["source"] == "CSDN" for item in payload["tech_items"])
    assert any(item["source"] == "Zhihu" for item in payload["search_links"])
    assert payload["directions"]
