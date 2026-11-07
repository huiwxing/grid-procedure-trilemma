"""Internet Archive access for the corpus rebuild (scripts/corpus/rebuild.py): capture URLs, the CDX index query and the
decoding of a capture's payload.

A manifest row is `verified_wayback` when the SHA-1 of the capture's bytes equals the CDX digest the archive recorded
at crawl time, and, where an earlier archive intake of the same captures (the manifest's provenance notes) recorded a
sha256, that matches too; anything else is `archive_copy_unverified` and never feeds a gold. The rebuild looks for the
capture whose bytes have the sha256 the manifest records.
"""
from __future__ import annotations

import gzip
import urllib.parse

WAYBACK = "https://web.archive.org/web/{timestamp}id_/{original}"
CDX_API = "https://web.archive.org/cdx/search/cdx?"
CDX_FIELDS = "timestamp,original,digest,length,statuscode,mimetype"


def cdx_query(target: str) -> str:
    """CDX API query: HTTP-200 captures of ``target``, one row per run of identical digests."""
    return CDX_API + urllib.parse.urlencode({"url": target, "output": "json", "fl": CDX_FIELDS,
                                             "filter": "statuscode:200", "collapse": "digest"})


GZIP_MAGIC = b"\x1f\x8b"


def decoded_payload(data: bytes) -> tuple[bytes, str]:
    """(document bytes, note): a payload archived with ``Content-Encoding: gzip`` is served gzip-encoded by
    ``id_`` URLs (the CDX digest is over the encoded bytes); decode it to get the PDF."""
    if data[:2] == GZIP_MAGIC:
        return gzip.decompress(data), "payload archived gzip-encoded, decoded"
    return data, ""
