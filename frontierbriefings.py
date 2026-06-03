"""Frontier Briefings — weekly non-consensus research pipeline.

Ingest manual/X/RSS news -> Opus 4.8 analyst pass (with vision)
-> push a formatted draft to Ghost Pro. Runs locally via `streamlit run`.
"""

import base64
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import jwt
import requests
import streamlit as st
import tweepy

# --- Config (edit these) ---------------------------------------------------
AGENT_NAME = "Waylam Smithers"
POST_TAG = "Frontier Briefings"
MAX_TWEETS = 50
MAX_TOKENS = 8192

# Selectable analyst engines: label -> (provider, model id). Cost rises top
# to bottom; bump the ids here when newer models ship.
MODELS = {
    "Opus 4.8 — Anthropic (best, $$$)": ("anthropic", "claude-opus-4-8"),
    "Sonnet 4.6 — Anthropic (balanced, $$)": ("anthropic", "claude-sonnet-4-6"),
    "Haiku 4.5 — Anthropic (cheapest, $)": ("anthropic", "claude-haiku-4-5-20251001"),
    "Gemini 3.5 Flash — Google (cheap, $)": ("google", "gemini-3.5-flash"),
}

# Each provider needs its own key in .env; only the selected one is required.
PROVIDER_KEY = {"anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}

# Analyst / macro accounts to pull from. Swap in your real targets.
X_ACCOUNTS = ["TrumpTruthOnX"]

HEADLINES_PER_FEED = 20

# Broad, keyless news scan that lets the analyst pick the topics. Reuters first
# (its native RSS is dead, so via Google News). Edit freely — order matters,
# Reuters leads the package.
RSS_FEEDS = {
    "Reuters": "https://news.google.com/rss/search?q=site:reuters.com+when:2d&hl=en-US&gl=US&ceid=US:en",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Google News Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
}

# Always needed regardless of model choice. The LLM provider key is added at
# runtime based on the selected engine.
BASE_REQUIRED = [
    "X_BEARER_TOKEN",
    "UNSPLASH_ACCESS_KEY",
    "GHOST_ADMIN_KEY",
    "GHOST_API_URL",
]

DISCLAIMER = (
    "<p><i><b>Disclaimer:</b> Frontier Lane provides this content for informational "
    "and educational purposes only, not as investment advice. Please read "
    'our full <a href="https://frontierlane.com/disclaimer">Disclaimer</a> for '
    "more information.</i></p>"
)

SYSTEM_PROMPT = """You are writing a short weekly note on behalf of an investment \
research firm, in the firm's collective voice, to a family-office client who trusts \
the firm's judgment. The team has deep buy-side experience and a knack for spotting \
shifts before the crowd.

Write from the consolidated intelligence the user provides (manual notes, X \
chatter, recent news headlines, and any chart screenshots).

PICK THE TOPICS YOURSELF. Scan everything and choose the 10 most genuinely \
interesting or important threads. Do not force a predetermined theme — follow what \
the news actually surfaces this week. Treat the user's pasted \
articles as your most reliable sources.

GO DEEPER THAN THE NEWS. Summarising is not the goal; the value is your thinking. Give \
each topic its own <h2> section and work through:
- the genuinely interesting angle — why it matters to now, what most people are missing
- the potential inflection point — what could change, and what would confirm it
- what it means for investors, and the second-order effects others won't have traced
- specific stocks (with tickers) and/or sectors that could benefit or suffer, and why
Interesting facts and data points are useful to prove your thinking and ideas.

How to write:
- LENGTH. About 1500 words TOTAL across all topics. With more topics, make \
each one tighter so the whole article stays in range. Substantial but still readable.
- PLAIN. Write like you're talking to a smart friend who isn't a specialist. \
Short sentences. Everyday words and numbers. Explain any technical term in a few words the \
first time you use it. No jargon walls, no acronym soup.
- VOICE. Write in the first person plural — "we", "our", "we're watching". Never use \
"I" or "my". This is the firm's shared view, but keep it warm and human, not stiff or \
corporate.
- HUMAN PROSE. Write in flowing but succinct paragraphs the way a thoughtful person \
actually writes a letter. Refer to specific datapoints. \
Let ideas connect and build. Calm, confident, personal: "here's what we're watching and why."
- LINK THE SOURCES. Some news headlines are followed by " | " and a URL; many have none. \
When you lean on a story that has a URL, hyperlink the few most relevant words in your \
sentence to it with an <a href="..."> tag. Only use URLs supplied with the headlines; \
never invent a link. Link sparingly, where it backs a claim, not on every sentence.
- NO DASHES. Never use en dashes or em dashes; they are a telltale sign of AI \
writing. Use commas, full stops, brackets, or a colon instead.
- SPELLING. Use British/Australian spelling throughout (e.g. realise, favour, \
centre, analyse, behaviour).
- CLOSE simply: what to watch next.

Return ONLY these tagged fields, nothing else:
<title>clear, plain headline (no jargon)</title>
<excerpt>1 sentence hook a non-expert understands</excerpt>
<meta_title>SEO title</meta_title>
<meta_description>SEO description under 155 chars</meta_description>
<body_html>the note as clean semantic HTML (h2, p, and inline <a> links only). No bold. No outer html/body tags.</body_html>"""


def log(msg):
    print(f"[{AGENT_NAME}] {msg}")


def load_env(required):
    """Native .env parse so we avoid a python-dotenv dependency."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        st.error(".env file not found. Copy the template and fill in your keys.")
        st.stop()

    env = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()

    missing = [k for k in required if not env.get(k) or env[k].endswith("...")]
    if missing:
        st.error(f"Missing keys in .env: {', '.join(missing)}")
        st.stop()
    return env


def fetch_tweets(bearer_token):
    """Last 7 days of high-relevance tweets from the watchlist."""
    client = tweepy.Client(bearer_token=bearer_token)
    accounts = " OR ".join(f"from:{a}" for a in X_ACCOUNTS)
    query = f"({accounts}) -is:retweet -is:reply lang:en"
    start = datetime.now(timezone.utc) - timedelta(days=7)

    resp = client.search_recent_tweets(
        query=query,
        start_time=start,
        max_results=min(MAX_TWEETS, 100),
        sort_order="relevancy",
        tweet_fields=["public_metrics", "created_at", "author_id"],
    )
    # tweepy types this call as a union (Response | requests.Response | dict),
    # so .data isn't statically guaranteed; getattr keeps the checker honest.
    tweets = getattr(resp, "data", None) or []
    if not tweets:
        log("X returned no tweets for the window.")
        return "No tweets found."

    lines = []
    for t in tweets:
        m = t["public_metrics"]
        engagement = m["like_count"] + m["retweet_count"] + m["reply_count"]
        lines.append(f"[{engagement} eng] {t['text']}")
    log(f"Pulled {len(lines)} tweets.")
    return "\n".join(lines)


def fetch_headlines():
    """Broad recent-news scan via RSS. Flaky feeds are skipped, not fatal.
    Returns (text, allowed_urls) — the URLs we handed the model, so any link it
    later invents can be stripped before publishing."""
    out = []
    allowed = set()
    for label, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            resp.raise_for_status()
            items = ET.fromstring(resp.content).findall(".//item")[:HEADLINES_PER_FEED]
        except (requests.RequestException, ET.ParseError) as e:
            log(f"Feed '{label}' failed, skipping: {e}")
            continue
        out.append(f"## {label}")
        for it in items:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            # Google News links are JS redirects that don't resolve on click; pass
            # those headlines title-only so the model never emits a dead link.
            clickable = link and "news.google.com" not in link
            if clickable:
                allowed.add(link)
            out.append(f"- {title} | {link}" if clickable else f"- {title}")
        log(f"{label}: {len(items)} headlines.")
    return "\n".join(out), allowed


def strip_unknown_links(html, allowed):
    """Unwrap any <a> the model invented: keep the visible words, drop the tag
    unless its href is a URL we actually supplied. Models fabricate plausible
    links (ft.com, wsj.com homepages) regardless of what the prompt forbids."""
    def keep_or_unwrap(m):
        return m.group(0) if m.group("href") in allowed else m.group("text")
    return re.sub(r'<a\b[^>]*\bhref="(?P<href>[^"]*)"[^>]*>(?P<text>.*?)</a>',
                  keep_or_unwrap, html, flags=re.DOTALL)


def encode_images(uploaded_files):
    """Provider-neutral (mime_type, base64) pairs for the chart screenshots."""
    imgs = [(f.type, base64.standard_b64encode(f.getvalue()).decode("utf-8"))
            for f in uploaded_files]
    if imgs:
        log(f"Encoded {len(imgs)} chart screenshot(s) for vision.")
    return imgs


def run_analysis(provider, model_id, api_key, package, images):
    """Dispatch the multimodal analysis to the selected provider."""
    runner = run_anthropic if provider == "anthropic" else run_gemini
    raw = runner(model_id, api_key, package, images)
    log(f"{model_id} analysis complete.")
    return raw


def run_anthropic(model_id, api_key, package, images):
    """System prompt is cached across runs to cut input cost."""
    client = anthropic.Anthropic(api_key=api_key)
    content: list = [{"type": "image",
                      "source": {"type": "base64", "media_type": m, "data": d}}
                     for m, d in images]
    content.append({"type": "text", "text": package})
    messages: list = [{"role": "user", "content": content}]

    resp = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def run_gemini(model_id, api_key, package, images):
    """Gemini via REST so we add no extra SDK dependency."""
    parts = [{"inline_data": {"mime_type": m, "data": d}} for m, d in images]
    parts.append({"text": package})

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        # thinkingBudget 0 stops Flash from spending the token budget reasoning
        # out loud, which was truncating the tagged answer.
        "generationConfig": {
            "maxOutputTokens": MAX_TOKENS,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    # The Anthropic SDK retries on its own; the bare REST call doesn't, so a
    # momentary overload (429/5xx) would otherwise sink the whole pipeline.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    for attempt in range(4):
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=120)
        if resp.status_code not in (429, 500, 503) or attempt == 3:
            break
        wait = 2 ** attempt
        log(f"Gemini {resp.status_code}, retrying in {wait}s ({attempt + 1}/3).")
        time.sleep(wait)
    resp.raise_for_status()
    cand = resp.json()["candidates"][0]
    answer = "".join(p["text"] for p in cand.get("content", {}).get("parts", [])
                      if "text" in p and not p.get("thought"))
    if not answer:
        raise RuntimeError(f"Gemini returned no text (finish: {cand.get('finishReason')}).")
    return answer


def extract(tag, text):
    """Tolerant tag parse: models sometimes garble the closing tag (e.g.
    </body>_html>), so fall back to a partial close match, then to end-of-text."""
    start = text.find(f"<{tag}>")
    if start == -1:
        return ""
    start += len(tag) + 2
    end = text.find(f"</{tag}>", start)
    if end == -1:
        end = text.find(f"</{tag[:4]}", start)
    return (text[start:end] if end != -1 else text[start:]).strip()


def unsplash_feature_image(access_key, topic):
    resp = requests.get(
        "https://api.unsplash.com/search/photos",
        headers={"Authorization": f"Client-ID {access_key}"},
        params={"query": topic, "per_page": 1, "orientation": "landscape"},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        log("Unsplash had no match; falling back to a generic finance query.")
        return unsplash_feature_image(access_key, "abstract finance technology")
    return results[0]["urls"]["regular"]


def inject_disclaimer(body_html):
    """Disclaimer is the first thing the reader sees below the hero image."""
    return DISCLAIMER + body_html


def push_to_ghost(admin_key, api_url, fields, feature_image_url):
    """Sign a short-lived JWT and POST an HTML draft to Ghost Admin."""
    key_id, secret = admin_key.split(":")
    now = int(time.time())
    token = jwt.encode(
        {"iat": now, "exp": now + 5 * 60, "aud": "/admin/"},
        bytes.fromhex(secret),
        algorithm="HS256",
        headers={"kid": key_id, "alg": "HS256"},
    )

    payload = {"posts": [{
        "title": fields["title"],
        "html": inject_disclaimer(fields["body_html"]) + "<hr>",
        "feature_image": feature_image_url,
        "custom_excerpt": fields["excerpt"],
        "meta_title": fields["meta_title"],
        "meta_description": fields["meta_description"],
        "status": "draft",
        "tags": [{"name": POST_TAG}],
    }]}

    resp = requests.post(
        f"{api_url}/ghost/api/admin/posts/?source=html",
        json=payload,
        headers={"Authorization": f"Ghost {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    post = resp.json()["posts"][0]
    post["url"] = f"{api_url}/ghost/#/editor/post/{post['id']}"
    log(f"Ghost draft created: {post['url']}")
    return post


def main():
    st.set_page_config(page_title="Frontier Briefings", page_icon="📈")
    st.title("📈 Frontier Briefings")
    st.caption(f"Non-consensus research pipeline — agent: {AGENT_NAME}")

    manual_feed = st.text_area(
        "Manual feed",
        height=240,
        placeholder="Paste premium news, paywalled excerpts (Bloomberg/WSJ/FT), "
        "URLs, and your own notes from the week here.",
    )
    uploads = st.file_uploader(
        "Supply-chain charts / diagrams (vision)",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=True,
    )

    model_label = st.selectbox("Analyst engine", list(MODELS))
    provider, model_id = MODELS[model_label]

    if not st.button("Run Analysis", type="primary"):
        return

    env = load_env(BASE_REQUIRED + [PROVIDER_KEY[provider]])
    log(f"Run started — engine: {model_id}")

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Pulling X watchlist...")
        tweets = fetch_tweets(env["X_BEARER_TOKEN"])

        st.write("Scanning news feeds...")
        headlines, allowed_links = fetch_headlines()

        st.write("Encoding screenshots...")
        images = encode_images(uploads)

        package = (
            f"=== MANUAL FEED ===\n{manual_feed or 'None provided.'}\n\n"
            f"=== X / TWITTER ===\n{tweets}\n\n"
            f"=== NEWS HEADLINES ===\n{headlines}"
        )

        st.write(f"Analyzing with {model_label}...")
        raw = run_analysis(provider, model_id,
                           env[PROVIDER_KEY[provider]], package, images)
        fields = {t: extract(t, raw) for t in
                  ["title", "excerpt", "meta_title", "meta_description", "body_html"]}
        if not fields["title"] or not fields["body_html"]:
            st.error("Model output was malformed. Raw response below.")
            st.code(raw)
            st.stop()
        fields["body_html"] = strip_unknown_links(fields["body_html"], allowed_links)

        st.write("Fetching feature image...")
        image_url = unsplash_feature_image(env["UNSPLASH_ACCESS_KEY"], fields["title"])

        st.write("Pushing draft to Ghost...")
        post = push_to_ghost(env["GHOST_ADMIN_KEY"], env["GHOST_API_URL"],
                             fields, image_url)
        status.update(label="Pipeline complete.", state="complete")

    st.success(f"Draft created: {post['title']}")
    st.markdown(f"[Open draft in Ghost]({post['url']})")
    st.image(image_url, caption="Feature image")
    st.subheader(fields["title"])
    st.write(f"*{fields['excerpt']}*")
    st.markdown(inject_disclaimer(fields["body_html"]), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
