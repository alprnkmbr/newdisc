"""
NewsDiscover RSS Bot
====================
Kurulum:
    pip install feedparser aiohttp scikit-learn numpy fastapi uvicorn

Çalıştırma:
    python rss_bot.py

Ardından frontend/index.html'i tarayıcıda aç.
API: http://localhost:8000/stories.json
"""

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import aiohttp.web
import feedparser
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
#  HABER KAYNAKLARI — 40+ RSS feed
# ─────────────────────────────────────────────
FEEDS = [
    # ── DÜNYA / WORLD ──
    {"name": "Al Jazeera English",  "url": "https://www.aljazeera.com/xml/rss/all.xml",              "cat": "World"},
    {"name": "BBC World",           "url": "http://feeds.bbci.co.uk/news/world/rss.xml",             "cat": "World"},
    {"name": "Reuters",             "url": "https://feeds.reuters.com/reuters/topNews",              "cat": "World"},
    {"name": "The Guardian World",  "url": "https://www.theguardian.com/world/rss",                  "cat": "World"},
    {"name": "AP News",             "url": "https://rsshub.app/apnews/topics/apf-topnews",           "cat": "World"},
    {"name": "DW English",          "url": "https://rss.dw.com/rdf/rss-en-all",                     "cat": "World"},
    {"name": "France 24",           "url": "https://www.france24.com/en/rss",                       "cat": "World"},
    {"name": "Euronews",            "url": "https://www.euronews.com/rss?format=mrss&level=theme&name=news","cat":"World"},

    # ── TÜRKÇE ──
    {"name": "BBC Türkçe",          "url": "https://feeds.bbci.co.uk/turkce/rss.xml",               "cat": "Türkiye"},
    {"name": "DW Türkçe",           "url": "https://rss.dw.com/rdf/rss-tur-all",                   "cat": "Türkiye"},
    {"name": "Al Jazeera Türkçe",   "url": "https://www.aljazeera.com.tr/feed",                    "cat": "Türkiye"},
    {"name": "Bianet",              "url": "https://bianet.org/bianet/rss",                        "cat": "Türkiye"},
    {"name": "T24",                 "url": "https://t24.com.tr/rss",                               "cat": "Türkiye"},
    {"name": "Anadolu Ajansı",      "url": "https://www.aa.com.tr/tr/rss/default?cat=gundem",      "cat": "Türkiye"},

    # ── TEKNOLOJİ ──
    {"name": "TechCrunch",          "url": "https://techcrunch.com/feed/",                         "cat": "Tech"},
    {"name": "The Verge",           "url": "https://www.theverge.com/rss/index.xml",               "cat": "Tech"},
    {"name": "Ars Technica",        "url": "http://feeds.arstechnica.com/arstechnica/index",       "cat": "Tech"},
    {"name": "Wired",               "url": "https://www.wired.com/feed/rss",                       "cat": "Tech"},
    {"name": "MIT Tech Review",     "url": "https://www.technologyreview.com/feed/",               "cat": "Tech"},
    {"name": "Webtekno",            "url": "https://www.webtekno.com/rss.xml",                     "cat": "Tech"},

    # ── BİLİM ──
    {"name": "Science Daily",       "url": "https://www.sciencedaily.com/rss/all.xml",             "cat": "Science"},
    {"name": "New Scientist",       "url": "https://www.newscientist.com/feed/home/",              "cat": "Science"},
    {"name": "Nature",              "url": "https://www.nature.com/nature.rss",                    "cat": "Science"},
    {"name": "NASA",                "url": "https://www.nasa.gov/rss/dyn/breaking_news.rss",       "cat": "Science"},

    # ── EKONOMİ ──
    {"name": "Financial Times",     "url": "https://www.ft.com/rss/home",                         "cat": "Business"},
    {"name": "Bloomberg Markets",   "url": "https://feeds.bloomberg.com/markets/news.rss",        "cat": "Business"},
    {"name": "Bloomberg HT",        "url": "https://www.bloomberght.com/feed",                    "cat": "Business"},
    {"name": "The Economist",       "url": "https://www.economist.com/latest/rss.xml",            "cat": "Business"},

    # ── POLİTİKA ──
    {"name": "Politico",            "url": "https://www.politico.com/rss/politics08.xml",         "cat": "Politics"},
    {"name": "Foreign Policy",      "url": "https://foreignpolicy.com/feed/",                     "cat": "Politics"},
    {"name": "The Hill",            "url": "https://thehill.com/feed/",                           "cat": "Politics"},

    # ── ÇEVRE ──
    {"name": "The Guardian Env.",   "url": "https://www.theguardian.com/environment/rss",         "cat": "Environment"},
    {"name": "Climate Home News",   "url": "https://www.climatechangenews.com/feed/",             "cat": "Environment"},
]

# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────
FETCH_INTERVAL_MINUTES = 30    # Kaç dakikada bir güncelle
MAX_ARTICLES_PER_FEED  = 15    # Feed başına max makale
MAX_STORIES            = 40    # Üretilecek max hikaye
SIMILARITY_THRESHOLD   = 0.20  # Kümeleme hassasiyeti (düşük = daha geniş kümeler)
OUTPUT_FILE            = Path("stories.json")
PORT                   = int(os.environ.get("PORT", 8000))  # Render PORT env var'ını kullanır

# ─────────────────────────────────────────────
#  RSS ÇEKME
# ─────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0 NewsDiscover/1.0 (RSS Bot; +https://github.com/newsdisc)"}

async def fetch_feed(session: aiohttp.ClientSession, feed: dict) -> list:
    articles = []
    try:
        async with session.get(feed["url"], headers=HEADERS, timeout=aiohttp.ClientTimeout(total=12)) as resp:
            if resp.status != 200:
                print(f"  ✗ {feed['name']}: HTTP {resp.status}")
                return []
            content = await resp.text(errors="replace")

        parsed = feedparser.parse(content)
        if not parsed.entries:
            print(f"  ✗ {feed['name']}: 0 entry (format sorunu?)")
            return []

        for entry in parsed.entries[:MAX_ARTICLES_PER_FEED]:
            art = parse_entry(entry, feed)
            if art:
                articles.append(art)

        print(f"  ✓ {feed['name']}: {len(articles)} makale")

    except asyncio.TimeoutError:
        print(f"  ✗ {feed['name']}: timeout")
    except Exception as e:
        print(f"  ✗ {feed['name']}: {e}")

    return articles


def parse_entry(entry, feed: dict) -> dict | None:
    title = getattr(entry, "title", "").strip()
    if not title or len(title) < 10:
        return None

    # İçerik — birkaç alanı dene
    content = ""
    for field in ["summary", "description", "content"]:
        val = getattr(entry, field, None)
        if isinstance(val, list):
            val = val[0].get("value", "") if val else ""
        if val:
            content = re.sub(r"<[^>]+>", " ", str(val))
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) > 40:
                break
    content = content[:1000]

    # URL
    url = getattr(entry, "link", "") or getattr(entry, "id", "")

    # Thumbnail
    thumbnail = None
    for attr in ["media_thumbnail", "media_content"]:
        items = getattr(entry, attr, [])
        if items and isinstance(items, list):
            thumbnail = items[0].get("url")
            break
    if not thumbnail and content:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
        if m:
            thumbnail = m.group(1)

    # Tarih
    pub_ts = time.time()
    for attr in ["published_parsed", "updated_parsed"]:
        val = getattr(entry, attr, None)
        if val:
            try:
                pub_ts = datetime(*val[:6], tzinfo=timezone.utc).timestamp()
            except Exception:
                pass
            break

    uid = hashlib.md5(url.encode()).hexdigest()[:12]

    return {
        "id": uid,
        "title": title,
        "content": content,
        "url": url,
        "source": feed["name"],
        "category": feed["cat"],
        "thumbnail": thumbnail,
        "ts": pub_ts,
        "domain": urlparse(url).netloc,
    }


async def fetch_all() -> list:
    connector = aiohttp.TCPConnector(limit=20, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_feed(session, f) for f in FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    for r in results:
        if isinstance(r, list):
            articles.extend(r)

    # Deduplicate by URL
    seen = set()
    unique = []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    return unique

# ─────────────────────────────────────────────
#  KÜMELENDİRME — TF-IDF + cosine similarity
# ─────────────────────────────────────────────
def cluster_articles(articles: list, threshold=SIMILARITY_THRESHOLD) -> list:
    if not articles:
        return []

    texts = [f"{a['title']} {a['title']} {a.get('content','')}" for a in articles]

    try:
        vec = TfidfVectorizer(max_features=800, min_df=1, sublinear_tf=True)
        matrix = vec.fit_transform(texts)
        sim = cosine_similarity(matrix)
    except Exception as e:
        print(f"  Vectorizer hatası: {e}")
        return [[a] for a in articles]

    parent = list(range(len(articles)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    rows, cols = np.where(sim >= threshold)
    for i, j in zip(rows, cols):
        if i < j:
            union(i, j)

    groups = {}
    for i, art in enumerate(articles):
        root = find(i)
        groups.setdefault(root, []).append(art)

    clusters = list(groups.values())

    # Önce çok kaynaklı ve büyük kümeleri getir
    clusters.sort(key=lambda c: (
        -len(set(a["source"] for a in c)),
        -len(c),
        -max(a["ts"] for a in c),
    ))

    return clusters


# ─────────────────────────────────────────────
#  HİKAYEYE DÖNÜŞTÜR
# ─────────────────────────────────────────────
def cluster_to_story(cluster: list, rank: int) -> dict:
    # En uzun içerikli makaleyi primary seç
    primary = max(cluster, key=lambda a: len(a.get("content", "")))

    # Kaynaklar (tekrar yok)
    sources = []
    seen_src = set()
    for a in sorted(cluster, key=lambda x: x["ts"], reverse=True):
        if a["source"] not in seen_src:
            sources.append({"name": a["source"], "url": a["url"], "domain": a["domain"]})
            seen_src.add(a["source"])

    # Kategori: çoğunluk oyu
    cat_count = {}
    for a in cluster:
        cat_count[a["category"]] = cat_count.get(a["category"], 0) + 1
    category = max(cat_count, key=cat_count.get)

    # En iyi özet: en uzun anlamlı içerik
    best_content = max(
        (a.get("content", "") for a in cluster),
        key=len, default=""
    )

    # Öne çıkan noktalar: farklı kaynaklardan ilk cümle
    key_points = []
    seen_kp = set()
    for art in cluster:
        if len(key_points) >= 4:
            break
        sents = re.split(r'[.!?]+', art.get("content", ""))
        for s in sents:
            s = s.strip()
            key = s[:30].lower()
            if len(s) > 50 and key not in seen_kp:
                seen_kp.add(key)
                key_points.append(s)
                break

    # Thumbnail
    thumbnail = next((a["thumbnail"] for a in cluster if a.get("thumbnail")), None)

    # Zaman
    ts = max(a["ts"] for a in cluster)

    # Önem skoru
    src_diversity = len(set(a["source"] for a in cluster))
    age_hours = (time.time() - ts) / 3600
    freshness = max(0, 72 - age_hours)
    importance = min(99, int(src_diversity * 18 + len(cluster) * 5 + freshness * 0.4))

    return {
        "id": primary["id"],
        "rank": rank + 1,
        "title": primary["title"],
        "summary": best_content[:500] if best_content else primary["title"],
        "key_points": key_points,
        "category": category,
        "sources": sources,
        "source_count": len(sources),
        "article_count": len(cluster),
        "thumbnail": thumbnail,
        "importance_score": importance,
        "ts": ts,
        "published_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "primary_url": primary["url"],
    }


# ─────────────────────────────────────────────
#  ANA DÖNGÜ
# ─────────────────────────────────────────────
async def run_once():
    print(f"\n{'─'*50}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Haberler çekiliyor...")

    articles = await fetch_all()
    print(f"\n  → {len(articles)} makale çekildi")

    articles = [a for a in articles if a.get("content") and len(a["content"]) > 40]
    print(f"  → {len(articles)} makale içerik filtresi geçti")

    clusters = cluster_articles(articles)
    print(f"  → {len(clusters)} küme oluşturuldu")

    stories = []
    for i, cluster in enumerate(clusters[:MAX_STORIES]):
        try:
            stories.append(cluster_to_story(cluster, i))
        except Exception as e:
            print(f"  ! Küme {i} hatası: {e}")

    output = {
        "stories": stories,
        "meta": {
            "total_articles": len(articles),
            "total_clusters": len(clusters),
            "sources_active": len(set(a["source"] for a in articles)),
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    }

    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ {len(stories)} hikaye → {OUTPUT_FILE}")
    return output


async def run_loop():
    while True:
        await run_once()
        print(f"\n  Sonraki güncelleme: {FETCH_INTERVAL_MINUTES} dakika sonra")
        await asyncio.sleep(FETCH_INTERVAL_MINUTES * 60)


# ─────────────────────────────────────────────
#  BASIT HTTP SUNUCU (CORS başlıklı)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  AIOHTTP WEB SUNUCU — thread yok, tek async loop
# ─────────────────────────────────────────────

async def handle_stories(request):
    if OUTPUT_FILE.exists():
        data = OUTPUT_FILE.read_bytes()
        return aiohttp.web.Response(
            body=data,
            content_type="application/json",
            charset="utf-8",
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=60"}
        )
    return aiohttp.web.Response(
        text=json.dumps({"status": "loading", "message": "Haberler çekiliyor, ~2 dakika bekleyin"}),
        content_type="application/json",
        status=503,
        headers={"Access-Control-Allow-Origin": "*", "Retry-After": "30"}
    )

async def handle_health(request):
    status = "ready" if OUTPUT_FILE.exists() else "loading"
    return aiohttp.web.Response(
        text=json.dumps({"status": status, "service": "NewsDiscover RSS Bot", "stories_ready": OUTPUT_FILE.exists()}),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"}
    )

async def handle_options(request):
    return aiohttp.web.Response(
        headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, OPTIONS"}
    )

# ─────────────────────────────────────────────
#  ANA DÖNGÜ — web sunucu + RSS fetch birlikte
# ─────────────────────────────────────────────

async def main():
    # Web uygulamasını kur
    app = aiohttp.web.Application()
    app.router.add_get("/stories.json", handle_stories)
    app.router.add_get("/health",       handle_health)
    app.router.add_get("/",             handle_health)
    app.router.add_options("/{tail:.*}", handle_options)

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    print(f"  🌐 Sunucu başladı: http://0.0.0.0:{PORT}")
    print(f"  📡 RSS çekme döngüsü başlıyor...")

    # RSS döngüsü
    while True:
        await run_once()
        print(f"  ⏰ Sonraki güncelleme: {FETCH_INTERVAL_MINUTES} dakika sonra")
        await asyncio.sleep(FETCH_INTERVAL_MINUTES * 60)

# ─────────────────────────────────────────────
#  BAŞLAT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  ⚡ NewsDiscover RSS Bot")
    print(f"  {len(FEEDS)} kaynak · her {FETCH_INTERVAL_MINUTES}dk güncelleme · port {PORT}")
    print("=" * 50)
    asyncio.run(main())
