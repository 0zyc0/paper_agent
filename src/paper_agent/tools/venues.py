from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from ..core.models import Paper


@dataclass
class VenueDecision:
    accepted: bool
    rank: str | None
    reason: str


class VenuePolicy:
    """Filters papers to arXiv, configured CCF A/B venues, and SCI Q1-Q3 journals.

    The bundled lists are seed lists for an MVP. Keep `config/venues.json` updated
    with the official CCF catalogue and the journal quartile source you choose.
    """

    def __init__(self, path: str | Path = "config/venues.json") -> None:
        self.path = Path(path)
        self.config = self._load_config()
        self.ccf_a = {_norm(item) for item in self.config.get("ccf_a_conferences", [])}
        self.ccf_b = {_norm(item) for item in self.config.get("ccf_b_conferences", [])}
        self.sci_q1_q3 = {_norm(item) for item in self.config.get("sci_q1_q3_journals", [])}
        self.aliases = {
            _norm(alias): _norm(target)
            for alias, target in self.config.get("aliases", {}).items()
        }

    def decide(self, paper: Paper) -> VenueDecision:
        if paper.arxiv_id or "arxiv" in paper.source.lower() or _norm(paper.venue) == "arxiv":
            if paper.fields_of_study and not _has_cs_arxiv_category(paper.fields_of_study):
                return VenueDecision(
                    False,
                    None,
                    f"Rejected because arXiv categories are not CS-related: {', '.join(paper.fields_of_study)}.",
                )
            return VenueDecision(True, "arXiv-CS", "Accepted because the paper is from arXiv with CS-compatible metadata.")
        if "google_scholar" in paper.source.lower():
            return VenueDecision(
                True,
                "Google-Scholar",
                "Accepted as a Google Scholar result; venue rank and bibliographic metadata should be verified.",
            )
        venue = self._canonical_venue(paper.venue)
        if not venue:
            return VenueDecision(False, None, "Rejected because venue metadata is missing.")
        if venue in self.ccf_a:
            return VenueDecision(True, "CCF-A", f"Accepted because venue matches configured CCF-A list: {paper.venue}.")
        if venue in self.ccf_b:
            return VenueDecision(True, "CCF-B", f"Accepted because venue matches configured CCF-B list: {paper.venue}.")
        if venue in self.sci_q1_q3:
            return VenueDecision(True, "SCI-Q1-Q3", f"Accepted because journal matches configured SCI Q1-Q3 list: {paper.venue}.")
        return VenueDecision(False, None, f"Rejected because venue is not in configured target list: {paper.venue}.")

    def filter(
        self,
        papers: list[Paper],
        *,
        target_venues: list[str] | None = None,
        target_venue_ranks: list[str] | None = None,
    ) -> list[Paper]:
        accepted: list[Paper] = []
        canonical_targets = {
            self._canonical_venue(target)
            for target in target_venues or []
            if target
        }
        rank_targets = set(target_venue_ranks or [])
        for paper in papers:
            decision = self.decide(paper)
            if decision.accepted and canonical_targets:
                paper_venue = self._canonical_venue(paper.venue)
                if paper_venue not in canonical_targets:
                    continue
            if decision.accepted and rank_targets:
                if decision.rank not in rank_targets:
                    continue
            if decision.accepted:
                paper.venue_rank = decision.rank
                paper.venue_reason = decision.reason
                accepted.append(paper)
        return accepted

    def venues_for_ranks(self, ranks: list[str] | None, area: str | None = None) -> list[str]:
        """Return a small venue set for rank-scoped DBLP searches."""
        rank_targets = set(ranks or [])
        venues: list[str] = []
        if "CCF-A" in rank_targets:
            venues.extend(self.config.get("ccf_a_conferences", []))
        if "CCF-B" in rank_targets:
            venues.extend(self.config.get("ccf_b_conferences", []))
        if "SCI-Q1-Q3" in rank_targets:
            venues.extend(self.config.get("sci_q1_q3_journals", []))
        if not venues:
            return []
        preferred = _area_preferred_venues(area)
        if preferred:
            preferred_keys = {_norm(item) for item in preferred}
            scoped = [venue for venue in venues if _norm(venue) in preferred_keys]
            if scoped:
                return list(dict.fromkeys(scoped))
        return list(dict.fromkeys(venues))[:14]

    def _canonical_venue(self, venue: str | None) -> str:
        value = _norm(venue)
        return self.aliases.get(value, value)

    def _load_config(self) -> dict:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return DEFAULT_VENUE_CONFIG


def _norm(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _has_cs_arxiv_category(categories: list[str]) -> bool:
    compatible_prefixes = ("cs.", "stat.ML", "eess.IV", "eess.SP")
    return any(category.startswith(compatible_prefixes) for category in categories)


def _area_preferred_venues(area: str | None) -> list[str]:
    key = _norm(area)
    mapping = {
        "nlp": ["ACL", "EMNLP", "COLING", "NAACL", "AAAI", "IJCAI"],
        "ai": ["AAAI", "IJCAI", "ICLR", "ICML", "NeurIPS", "KDD", "WWW"],
        "ml": ["ICML", "NeurIPS", "ICLR", "KDD", "AAAI", "IJCAI", "UAI"],
        "cv": ["CVPR", "ICCV", "ECCV", "AAAI", "ICLR", "ICML"],
        "db": ["SIGMOD", "VLDB", "ICDE", "KDD", "CIKM", "WSDM", "PODS"],
        "se": ["ICSE", "ASE", "ESEC/FSE", "ISSTA", "Software: Practice and Experience"],
        "security": ["CCS", "USENIX Security", "NDSS", "S&P", "TIFS", "Computers & Security"],
        "systems": ["OSDI", "SOSP", "ASPLOS", "ISCA", "HPCA", "MICRO", "FAST"],
        "networks": ["SIGCOMM", "INFOCOM", "MobiCom", "CoNEXT", "IMC", "MobiSys"],
        "hci": ["CHI", "CSCW", "UbiComp"],
        "graphics": ["SIGGRAPH"],
        "robotics": ["ICRA", "AAAI", "IJCAI"],
    }
    return mapping.get(key, [])


DEFAULT_VENUE_CONFIG = {
    "ccf_a_conferences": [
        "AAAI",
        "ACL",
        "ASPLOS",
        "CCS",
        "CHI",
        "CVPR",
        "DAC",
        "FOCS",
        "HPCA",
        "ICCV",
        "ICDE",
        "ICML",
        "ICSE",
        "IJCAI",
        "INFOCOM",
        "ISCA",
        "KDD",
        "MobiCom",
        "NeurIPS",
        "OSDI",
        "PLDI",
        "POPL",
        "SIGCOMM",
        "SIGGRAPH",
        "SIGIR",
        "SIGMOD",
        "SOSP",
        "STOC",
        "USENIX Security",
        "VLDB",
        "WWW",
    ],
    "ccf_b_conferences": [
        "ASE",
        "CIKM",
        "COLING",
        "CoNEXT",
        "CSCW",
        "DSN",
        "ECCV",
        "EMNLP",
        "ESEC/FSE",
        "FAST",
        "ICLR",
        "ICME",
        "ICRA",
        "ICWSM",
        "IMC",
        "ISSTA",
        "MICRO",
        "MobiSys",
        "NAACL",
        "NDSS",
        "PODS",
        "RECOMB",
        "S&P",
        "UbiComp",
        "UAI",
        "WSDM",
    ],
    "sci_q1_q3_journals": [
        "ACM Computing Surveys",
        "Artificial Intelligence",
        "Computers & Security",
        "Data Mining and Knowledge Discovery",
        "IEEE Transactions on Information Forensics and Security",
        "IEEE Transactions on Knowledge and Data Engineering",
        "IEEE Transactions on Neural Networks and Learning Systems",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "Information Fusion",
        "Information Sciences",
        "Journal of Machine Learning Research",
        "Machine Learning",
        "Pattern Recognition",
        "Science China Information Sciences",
        "Software: Practice and Experience",
        "The VLDB Journal",
    ],
    "aliases": {
        "NIPS": "NeurIPS",
        "Neural Information Processing Systems": "NeurIPS",
        "Conference on Neural Information Processing Systems": "NeurIPS",
        "Proceedings of the AAAI Conference on Artificial Intelligence": "AAAI",
        "Annual Meeting of the Association for Computational Linguistics": "ACL",
        "Empirical Methods in Natural Language Processing": "EMNLP",
        "IEEE Symposium on Security and Privacy": "S&P",
        "IEEE S&P": "S&P",
        "Oakland": "S&P",
        "The Web Conference": "WWW",
        "World Wide Web Conference": "WWW",
        "International Conference on Learning Representations": "ICLR",
        "International Conference on Machine Learning": "ICML",
        "Computer Vision and Pattern Recognition": "CVPR",
        "T-PAMI": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "TPAMI": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "IEEE Trans. Pattern Anal. Mach. Intell.": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "TKDE": "IEEE Transactions on Knowledge and Data Engineering",
        "IEEE Trans. Knowl. Data Eng.": "IEEE Transactions on Knowledge and Data Engineering",
        "TIFS": "IEEE Transactions on Information Forensics and Security",
        "IEEE Trans. Inf. Forensics Secur.": "IEEE Transactions on Information Forensics and Security",
        "TNNLS": "IEEE Transactions on Neural Networks and Learning Systems",
        "IEEE Trans. Neural Networks Learn. Syst.": "IEEE Transactions on Neural Networks and Learning Systems",
    },
}
