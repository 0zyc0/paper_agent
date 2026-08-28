from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import os
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

from .models import Paper, normalize_text, utc_now_iso
from .rank import rank_papers
from ..tools.http_client import get_text
from ..tools.search import ArxivClient, DblpClient, OpenAlexClient, dedupe_papers

try:
    from .. import local_config
except ImportError:  # pragma: no cover - local config is optional
    local_config = None


@dataclass
class DiscoveryItem:
    title: str
    source: str
    url: str
    summary: str = ""
    published_at: str = ""
    kind: str = "article"

    def to_dict(self) -> dict:
        return asdict(self)


class DiscoveryService:
    def __init__(self) -> None:
        self.openalex = OpenAlexClient()
        self.arxiv = ArxivClient()
        self.dblp = DblpClient()

    def discover(self, topic: str, *, recent_years: int = 1, limit: int = 18) -> dict:
        return self.discover_profile({"primary_topic": topic, "topics": [topic], "seed_titles": [], "seed_abstracts": []}, recent_years=recent_years, limit=limit)

    def discover_profile(self, profile: dict, *, recent_years: int = 1, limit: int = 24) -> dict:
        topic = normalize_text(str(profile.get("primary_topic") or "")) or "computer science"
        display_topic = normalize_text(str(profile.get("display_topic") or "")) or topic
        topics = _clean_topics(profile.get("topics") or [topic])
        current_year = datetime.now(timezone.utc).year
        from_year = current_year - max(1, recent_years) + 1
        source_status: list[dict] = []
        papers: list[Paper] = []
        paper_groups: dict[str, list[dict]] = {}

        per_topic_limit = max(6, min(10, limit // max(1, min(len(topics), 3))))
        for query_topic in topics[:3]:
            topic_papers: list[Paper] = []
            for name, fetch in [
                ("OpenAlex", lambda q=query_topic: self.openalex.search(q, limit=per_topic_limit, from_year=from_year, to_year=current_year)),
                ("arXiv", lambda q=query_topic: self.arxiv.search(q, limit=per_topic_limit)),
                ("DBLP", lambda q=query_topic: self.dblp.search(q, limit=per_topic_limit, from_year=from_year, to_year=current_year)),
            ]:
                try:
                    found = fetch()
                    papers.extend(found)
                    topic_papers.extend(found)
                    source_status.append({"source": name, "topic": query_topic, "status": "ok", "count": len(found), "error": ""})
                except Exception as exc:
                    source_status.append({"source": name, "topic": query_topic, "status": "error", "count": 0, "error": str(exc)[:300]})
            paper_groups[query_topic] = [_paper_to_discovery_payload(paper) for paper in rank_papers(dedupe_papers(topic_papers), query_topic, limit=8)]

        ranked = rank_papers(dedupe_papers(papers), " ".join(topics), limit=limit)
        rss_items, rss_status = self._rss_items(topic)
        source_status.extend(rss_status)
        search_links = _platform_search_links(topic)
        return {
            "topic": topic,
            "display_topic": display_topic,
            "topics": topics,
            "profile": {
                "primary_topic": topic,
                "display_topic": display_topic,
                "topics": topics,
                "seed_titles": list(profile.get("seed_titles") or [])[:8],
                "paper_count": int(profile.get("paper_count") or 0),
            },
            "updated_at": utc_now_iso(),
            "sources": source_status,
            "papers": [_paper_to_discovery_payload(paper) for paper in ranked],
            "paper_groups": paper_groups,
            "tech_items": [item.to_dict() for item in rss_items[:12]],
            "search_links": [item.to_dict() for item in search_links],
            "trends": _trend_cards(topic, ranked, rss_items, topics=topics),
            "directions": _direction_cards(topic, ranked, profile),
        }

    def _rss_items(self, topic: str) -> tuple[list[DiscoveryItem], list[dict]]:
        feeds = _configured_rss_feeds(topic)
        items: list[DiscoveryItem] = []
        statuses: list[dict] = []
        for feed in feeds:
            name = feed["name"]
            url = feed["url"]
            try:
                parsed = _parse_rss(url, source=name)
                items.extend(parsed)
                statuses.append({"source": name, "status": "ok", "count": len(parsed), "error": ""})
            except Exception as exc:
                statuses.append({"source": name, "status": "error", "count": 0, "error": str(exc)[:300]})
        return items, statuses


def _configured_rss_feeds(topic: str) -> list[dict[str, str]]:
    configured = getattr(local_config, "DISCOVERY_RSS_FEEDS", []) if local_config else []
    feeds: list[dict[str, str]] = []
    for item in configured or []:
        if isinstance(item, dict) and item.get("url"):
            feeds.append({"name": str(item.get("name") or "RSS"), "url": str(item["url"]).format(query=quote_plus(topic))})
    rsshub_base = os.getenv("RSSHUB_BASE_URL") or (getattr(local_config, "RSSHUB_BASE_URL", "") if local_config else "")
    if rsshub_base:
        base = rsshub_base.rstrip("/")
        query = quote_plus(topic)
        feeds.extend(
            [
                {"name": "CSDN RSSHub", "url": f"{base}/csdn/article/search/{query}"},
                {"name": "Zhihu RSSHub", "url": f"{base}/zhihu/search/{query}"},
            ]
        )
    return feeds


def _clean_topics(values) -> list[str]:
    topics: list[str] = []
    for value in values or []:
        topic = normalize_text(str(value))
        if topic and topic.lower() not in {item.lower() for item in topics}:
            topics.append(topic)
    return topics[:5] or ["computer science"]


def _parse_rss(url: str, *, source: str) -> list[DiscoveryItem]:
    text = get_text(url, timeout=20)
    root = ET.fromstring(text)
    items = []
    for node in root.findall(".//item")[:20]:
        title = _rss_text(node, "title")
        link = _rss_text(node, "link")
        if not title or not link:
            continue
        items.append(
            DiscoveryItem(
                title=title,
                source=source,
                url=link,
                summary=_rss_text(node, "description")[:260],
                published_at=_rss_text(node, "pubDate"),
                kind="article",
            )
        )
    return items


def _rss_text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    return normalize_text("".join(child.itertext())) if child is not None else ""


def _platform_search_links(topic: str) -> list[DiscoveryItem]:
    query = quote_plus(topic)
    return [
        DiscoveryItem("微信公众号相关文章搜索", "Sogou WeChat", f"https://weixin.sogou.com/weixin?type=2&query={query}", kind="search"),
        DiscoveryItem("CSDN 技术文章搜索", "CSDN", f"https://so.csdn.net/so/search?q={query}", kind="search"),
        DiscoveryItem("知乎讨论搜索", "Zhihu", f"https://www.zhihu.com/search?type=content&q={query}", kind="search"),
        DiscoveryItem("GitHub 相关项目搜索", "GitHub", f"https://github.com/search?q={query}&type=repositories", kind="search"),
    ]


def _paper_to_discovery_payload(paper: Paper) -> dict:
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract or "",
        "year": paper.year,
        "published_at": paper.published_at,
        "venue": paper.venue,
        "source": paper.source,
        "url": paper.source_url or paper.pdf_url or "",
        "citation_count": paper.citation_count,
    }


def _trend_cards(topic: str, papers: list[Paper], items: list[DiscoveryItem], *, topics: list[str]) -> list[dict]:
    source_counts: dict[str, int] = {}
    venue_counts: dict[str, int] = {}
    for paper in papers:
        source_counts[paper.source] = source_counts.get(paper.source, 0) + 1
        if paper.venue:
            venue_counts[paper.venue] = venue_counts.get(paper.venue, 0) + 1
    top_venues = ", ".join(name for name, _ in sorted(venue_counts.items(), key=lambda item: item[1], reverse=True)[:4])
    return [
        {
            "label": "最新论文",
            "title": f"{topic} 的近期论文增量",
            "summary": f"基于 {len(topics)} 个历史研究方向发现 {len(papers)} 篇近期论文，主要来源为 {', '.join(source_counts) or '待获取'}。",
        },
        {
            "label": "活跃 venue",
            "title": top_venues or "等待更多会议/期刊信号",
            "summary": "可优先查看这些 venue 中的新论文，判断该方向是否进入顶会/期刊热点。",
        },
        {
            "label": "技术社区",
            "title": f"{len(items)} 条技术动态可补充阅读",
            "summary": "公众号、CSDN、知乎更适合跟踪工程解读和中文讨论，不能替代论文证据。",
        },
    ]


def _direction_cards(topic: str, papers: list[Paper], profile: dict) -> list[dict]:
    if not papers:
        return [{"label": "待检索", "text": f"先获取 {topic} 的近期论文，再自动生成可验证方向。"}]
    recent_titles = "; ".join(paper.title[:80] for paper in papers[:3])
    seed_count = len(profile.get("seed_titles") or [])
    return [
        {"label": "综述切入", "text": f"结合当前项目 {seed_count} 篇种子论文和近期动态，梳理 {topic} 的方法分支。代表新论文：{recent_titles}。"},
        {"label": "实验切入", "text": "优先比较近期论文使用的数据集、指标和开源代码，寻找尚未统一评测的问题。"},
        {"label": "问题切入", "text": "筛选只有元数据但缺少摘要/全文的高相关论文，后续补全文以验证真实贡献。"},
    ]
