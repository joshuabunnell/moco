#!/usr/bin/env python3
"""
Refreshes ~/sol-docs/ from the live ASU Research Computing docs site
(docs.rc.asu.edu). See ../SKILL.md for why this exists and how to use it.

Pure stdlib + the `pandoc` binary (already present at /usr/bin/pandoc on
Sol). No pip installs, no API keys.

Usage:
    python3 refresh.py                 # full run, writes to ~/sol-docs
    python3 refresh.py --dry-run       # report what would change, write nothing
    python3 refresh.py --limit 10      # only process the first N pages (testing)
    python3 refresh.py --verbose       # print each page as it's processed
"""
import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser

BASE = "https://docs.rc.asu.edu"
SITEMAP_URL = f"{BASE}/sitemap.xml"
OUTPUT_DIR = os.path.expanduser("~/sol-docs")
INDEX_PATH = os.path.join(OUTPUT_DIR, "INDEX.md")
USER_AGENT = "sol-docs-refresh-skill/1.0 (personal ASU RC docs cache; contact via ASURITE jpbunnel)"
REQUEST_DELAY_SECONDS = 0.3

# Blog/feed-style sections that Docusaurus generates alongside the real docs
# (changelog entries, event announcements, tag/author index pages, search).
# These aren't reference material and churn constantly, so they're excluded
# rather than treated as "docs that changed."
EXCLUDE_PREFIXES = ("/changelog", "/events", "/news", "/search")
EXCLUDE_SUBSTRINGS = ("/tags/", "/authors")

# Files in ~/sol-docs/ that this script does not own and must never touch,
# even though they live in the same directory. policies.md lives on a
# different domain entirely; home.md is a hand-curated summary of the
# landing page, which is a bespoke hero/marketing layout with no content
# div to extract generically the way every other page has.
PROTECTED_FILES = {"INDEX.md", "policies.md", "home.md"}

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

# Content wrapper classes to look for, in priority order. Docusaurus doc
# pages use theme-doc-markdown; auto-generated category/index pages (like
# /accounts or /computing) use a different template with no markdown div.
CONTENT_CLASS_CANDIDATES = ["theme-doc-markdown", "generatedIndexPage"]


class ClassExtractor(HTMLParser):
    """Pulls the inner HTML of the first element whose class list contains
    one of `target_classes`, tracking tag depth so nested divs/articles
    inside (admonitions, tab panels, etc.) don't confuse the boundary."""

    def __init__(self, target_classes):
        super().__init__(convert_charrefs=False)
        self.target_classes = target_classes
        self.depth = 0
        self.capture_at_depth = None
        self.out = []
        self.matched_class = None

    def _classes(self, attrs):
        for k, v in attrs:
            if k == "class":
                return (v or "").split()
        return []

    def handle_starttag(self, tag, attrs):
        is_void = tag in VOID_ELEMENTS
        if isinstance(self.capture_at_depth, int):
            self.out.append(self.get_starttag_text())
        elif self.matched_class is None:
            classes = self._classes(attrs)
            for c in self.target_classes:
                if any(cls == c or cls.startswith(c + "_") for cls in classes):
                    self.matched_class = c
                    if not is_void:
                        self.capture_at_depth = self.depth + 1
                    break
        if not is_void:
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        if isinstance(self.capture_at_depth, int):
            self.out.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag in VOID_ELEMENTS:
            return
        if isinstance(self.capture_at_depth, int):
            if self.depth == self.capture_at_depth:
                self.capture_at_depth = "done"
            else:
                self.out.append(f"</{tag}>")
        self.depth -= 1

    def handle_data(self, data):
        if isinstance(self.capture_at_depth, int):
            self.out.append(data)

    def handle_entityref(self, name):
        if isinstance(self.capture_at_depth, int):
            self.out.append(f"&{name};")

    def handle_charref(self, name):
        if isinstance(self.capture_at_depth, int):
            self.out.append(f"&#{name};")

    def result(self):
        return "".join(self.out) if self.out else None


class TabRewriter(HTMLParser):
    """Docusaurus tab groups (`:::tabs` in the source MDX, e.g. the
    per-role blocks on /voyager-accounts or per-OS install instructions
    elsewhere on the site) render as a `role="tablist"` of labels followed
    by one `role="tabpanel"` div per label, in the same order — with CSS
    `hidden` controlling which one shows. Flattened straight to markdown,
    all panels run together with no indication which label they belong to,
    which is actively misleading wherever tabs distinguish content (OS,
    role, client) rather than just alternative phrasing.

    This does a full pass-through rewrite of the HTML, re-emitting every
    event unchanged except: it collects each tablist's `<li>` label text,
    then injects a synthetic `<h3>{label}</h3>` as the first child of the
    next unlabeled tabpanel, consuming labels in document order. Multiple
    tab groups on one page work correctly as long as each group's panels
    appear before the next group's tablist, which is how Docusaurus
    always renders them."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out = []
        self.in_tablist_depth = None
        self.in_label_depth = None
        self.label_buf = []
        self.pending_labels = []
        self.label_index = 0
        self.depth = 0
        self.awaiting_panel_label = False

    @staticmethod
    def _attr(attrs, name):
        for k, v in attrs:
            if k == name:
                return v
        return None

    def handle_starttag(self, tag, attrs):
        is_void = tag in VOID_ELEMENTS
        if self._attr(attrs, "role") == "tablist":
            self.in_tablist_depth = self.depth + 1
            self.pending_labels = []
            self.label_index = 0
        elif self.in_tablist_depth is not None and tag == "li":
            self.in_label_depth = self.depth + 1
            self.label_buf = []
        elif (
            self._attr(attrs, "role") == "tabpanel"
            and self.label_index < len(self.pending_labels)
        ):
            self.awaiting_panel_label = True
        self.out.append(self.get_starttag_text())
        if self.awaiting_panel_label:
            label = self.pending_labels[self.label_index]
            self.label_index += 1
            self.out.append(f"<h3>{html.escape(label)}</h3>")
            self.awaiting_panel_label = False
        if not is_void:
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        self.out.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag not in VOID_ELEMENTS:
            self.depth -= 1
        if self.in_label_depth is not None and self.depth == self.in_label_depth - 1:
            self.pending_labels.append("".join(self.label_buf).strip())
            self.in_label_depth = None
        if self.in_tablist_depth is not None and self.depth == self.in_tablist_depth - 1:
            self.in_tablist_depth = None
        if tag not in VOID_ELEMENTS:
            self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if self.in_label_depth is not None:
            self.label_buf.append(data)
        self.out.append(data)

    def handle_entityref(self, name):
        if self.in_label_depth is not None:
            self.label_buf.append(f"&{name};")
        self.out.append(f"&{name};")

    def handle_charref(self, name):
        if self.in_label_depth is not None:
            self.label_buf.append(f"&#{name};")
        self.out.append(f"&#{name};")

    def result(self):
        return "".join(self.out)


def rewrite_tabs(fragment_html):
    parser = TabRewriter()
    parser.feed(fragment_html)
    return parser.result()


def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def discover_urls():
    """Returns a sorted list of (path, url) pairs for real doc pages,
    derived from the site's sitemap.xml (no lastmod dates are published,
    so there's no cheaper way to know what changed upstream than fetching)."""
    xml = fetch(SITEMAP_URL)
    locs = re.findall(r"<loc>([^<]+)</loc>", xml)
    pairs = []
    for loc in locs:
        loc = html.unescape(loc)
        if not loc.startswith(BASE):
            continue
        path = loc[len(BASE):] or "/"
        if path == "/":
            continue  # bespoke landing page, see PROTECTED_FILES note on home.md
        if any(path.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if any(s in path for s in EXCLUDE_SUBSTRINGS):
            continue
        pairs.append((path, loc))
    pairs.sort()
    return pairs


def path_to_filename(path):
    p = path.strip("/")
    if p == "":
        return "home.md"
    return p.replace("/", "-") + ".md"


def extract_title(page_html):
    m = re.search(r"<title[^>]*>([^<]*)</title>", page_html)
    return html.unescape(m.group(1)).strip() if m else "Untitled"


def extract_content_html(page_html):
    parser = ClassExtractor(CONTENT_CLASS_CANDIDATES)
    parser.feed(page_html)
    fragment = parser.result()
    return fragment, parser.matched_class


def _pandoc_atx_flag():
    """Pandoc renamed --atx-headers to --markdown-headings=atx in 2.11.4.
    Sol's system pandoc (2.0.6, as of this writing) needs the old name;
    detect once so the skill keeps working across a `module load` pandoc
    or an eventual OS upgrade."""
    help_text = subprocess.run(
        ["pandoc", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.decode(errors="replace")
    return ["--markdown-headings=atx"] if "--markdown-headings" in help_text else ["--atx-headers"]


BREADCRUMB_RE = re.compile(r'<nav\b[^>]*aria-label="Breadcrumbs"[^>]*>.*?</nav>', re.DOTALL)
# Every page ends with a prev/next "Passwordless SSH keys -> Getting Access"
# style pager. It's page-flip chrome for the site's own reading order, not
# reference content, and (like the card links) wraps its label in a <div>
# pandoc can't turn into an inline link, so it'd otherwise leave dangling
# `[](url)` / "Previous" / "Title" fragments at the end of every file.
PAGINATION_NAV_RE = re.compile(r'<nav\b[^>]*class="pagination-nav[^"]*"[^>]*>.*?</nav>', re.DOTALL)
CODE_LANG_RE = re.compile(r'\.language-([a-zA-Z0-9_+-]+)')


def html_to_markdown(fragment_html, atx_flag):
    # Docusaurus nests the breadcrumb trail (Accounts > Voyager Account
    # Manager > ...) inside the content wrapper on category-index pages
    # (e.g. /accounts), even though it's a sibling on normal doc pages.
    # Either way it's chrome, not content, so drop it before conversion.
    fragment_html = BREADCRUMB_RE.sub("", fragment_html)
    fragment_html = PAGINATION_NAV_RE.sub("", fragment_html)
    fragment_html = rewrite_tabs(fragment_html)
    fragment_html = re.sub(r"<img\b[^>]*>", "", fragment_html)
    proc = subprocess.run(
        # -simple_tables+pipe_tables: pandoc's default table writer picks a
        # layout per-table (simple/grid/pipe) based on cell content, which
        # produces the un-diffable, hard-to-hand-edit grid style for a lot
        # of the hardware-spec/partition tables on this site. Pipe tables
        # match every hand-authored table already in sol-docs.
        ["pandoc", "-f", "html", "-t", "markdown-simple_tables+pipe_tables"]
        + atx_flag
        + ["--wrap=none"],
        input=fragment_html.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pandoc failed: {proc.stderr.decode(errors='replace')[:500]}")
    return proc.stdout.decode("utf-8", errors="replace")


# Docusaurus's own layout markup (admonition boxes, anchor permalinks, CSS
# module classes) rides along as pandoc fenced-div / attribute syntax that
# looks exactly like `{...}` — the same shape as shell parameter expansion
# (`${VAR}`) and awk/sed program blocks (`'{print $1}'`) that show up
# verbatim inside code samples on the software-usage pages (abaqus, gromacs,
# vasp, etc.). Cleaning must never touch fenced code blocks, or it silently
# corrupts every code sample on those pages.
CODE_FENCE_SPLIT_RE = re.compile(r"(^[ \t]*```.*?\n.*?^[ \t]*```[ \t]*$)", re.MULTILINE | re.DOTALL)

ADMONITION_RE = re.compile(
    r"::: \{\.admonitionHeading[^}]*\}\s*\n(\w+)\s*\n:::", re.MULTILINE
)
HASH_LINK_RE = re.compile(r"\[\]\([^)]*\)\{\.hash-link\}")
ATTR_SPAN_RE = re.compile(r"\{[^{}]*\}")
BARE_FENCE_RE = re.compile(r"^[ \t]*:{3,}[ \t]*$", re.MULTILINE)
ESCAPED_PUNCT_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!\"'>~^$%&|<>])")
HEADING_RE = re.compile(r"^(#{1,6})(\s+)", re.MULTILINE)
BLANK_RUN_RE = re.compile(r"\n{3,}")
# Category-index pages (/accounts, /connecting, /computing, ...) render each
# card as a single <a> wrapping both its heading and description. Pandoc
# can't express a block-wrapping link inline, so it drops the bare href as
# a dangling `[](url)` right before the heading it belonged to. Re-attach it
# to the heading text instead of leaving an orphaned empty link sitting there.
CARD_LINK_RE = re.compile(r"\[\]\((/[^)\s]+)\)(?:\{[^}]*\})?\s*\n+(#{1,6})(\s+)(.*)")
# Numbered how-to steps that break for a screenshot mid-sequence wrap the
# image in a bare <div> and splice in an empty <!-- --> comment so the list
# numbering continues across the break. Images are already dropped (this
# cache is text-only), which leaves these as orphaned raw-HTML lines with
# nothing left to wrap or continue.
ORPHAN_HTML_LINE_RE = re.compile(r"^[ \t]*(<div>|</div>|<!-- -->)[ \t]*$", re.MULTILINE)


def _clean_prose(text):
    text = ADMONITION_RE.sub(lambda m: f"**{m.group(1).capitalize()}:**", text)
    text = HASH_LINK_RE.sub("", text)
    text = CARD_LINK_RE.sub(lambda m: f"{m.group(2)}{m.group(3)}[{m.group(4).rstrip()}]({m.group(1)})", text)
    text = ATTR_SPAN_RE.sub("", text)
    text = BARE_FENCE_RE.sub("", text)
    text = ORPHAN_HTML_LINE_RE.sub("", text)
    text = ESCAPED_PUNCT_RE.sub(r"\1", text)
    text = HEADING_RE.sub(r"#\1\2", text)  # demote: page's H1 becomes our H2, etc.
    return text


CODE_FENCE_OPEN_RE = re.compile(r"^([ \t]*```)\s*\{[^}]*\}", re.MULTILINE)


def _simplify_fence_open(match):
    indent = match.group(1)
    lang = CODE_LANG_RE.search(match.group(0))
    return f"{indent}{lang.group(1)}" if lang else indent


def clean_markdown(md):
    md = md.replace("​", "")
    parts = CODE_FENCE_SPLIT_RE.split(md)
    md = "".join(
        part if i % 2 else _clean_prose(part) for i, part in enumerate(parts)
    )
    # Pandoc carries the syntax-highlighter's full class/style soup onto the
    # fence line (e.g. `{.prism-code .language-bash ... style="..."}`); this
    # part of the fence *is* metadata, not code, so it's fine to simplify
    # even though we otherwise never touch inside a fenced block.
    md = CODE_FENCE_OPEN_RE.sub(_simplify_fence_open, md)
    lines = [line.rstrip() for line in md.split("\n")]
    md = "\n".join(lines)
    md = BLANK_RUN_RE.sub("\n\n", md)
    return md.strip()


def build_file_content(url, title, body_md):
    return f"<!-- source: {url} -->\n# {title}\n\n{body_md}\n"


def process_page(path, url, atx_flag, verbose=False):
    # The site 308-redirects any non-trailing-slash route to its trailing-
    # slash form (Cloudflare-fronted static host), which urllib on this
    # system's Python doesn't auto-follow (308 support landed in a newer
    # stdlib than what's installed here). Fetch the canonical form directly;
    # `url` itself (no trailing slash) is still what gets recorded as the
    # page's source, matching the existing sol-docs file convention.
    fetch_url = url if url.endswith("/") else url + "/"
    page_html = fetch(fetch_url)
    title = extract_title(page_html)
    fragment, matched = extract_content_html(page_html)
    if fragment is None:
        return None, f"no known content wrapper found (tried {CONTENT_CLASS_CANDIDATES})"
    raw_md = html_to_markdown(fragment, atx_flag)
    body = clean_markdown(raw_md)
    if not body:
        return None, "extracted body was empty after cleaning"
    content = build_file_content(url, title, body)
    if verbose:
        print(f"    matched .{matched}, {len(body)} chars", file=sys.stderr)
    return (content, title), None


def load_existing(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def write_file(filename, content, dry_run):
    if not dry_run:
        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(content)


def rebuild_index(entries, run_date, stats, dry_run):
    """entries: list of (filename, url, title), sorted by filename."""
    lines = [
        "# Index: ASU RC / Sol Documentation Cache",
        "",
        "Local markdown mirror of docs.rc.asu.edu (the ASU RC documentation site — "
        "confirmed as the correct source by cross-referencing ASU RC's own account-"
        "management emails, which link directly to pages under this domain). "
        "Kept in sync by the `sol-docs-refresh` skill. Docs can change upstream; "
        "re-run the skill any time you want the cache current.",
        "",
        f"> **Last refresh:** {run_date} — "
        f"{stats['added']} added, {stats['updated']} updated, "
        f"{stats['removed']} removed, {stats['unchanged']} unchanged "
        f"({stats['failed']} fetch failures kept their previous cached copy).",
        "",
        "`policies.md` (the ASU RC acceptable-use policy) lives on a different "
        "domain — cores.research.asu.edu, not docs.rc.asu.edu — and is out of "
        "scope for this skill; it's a one-time reference copy, update it by hand "
        "if it ever changes.",
        "",
        "| File | Source | Title |",
        "|---|---|---|",
    ]
    for filename, url, title in entries:
        lines.append(f"| `{filename}` | {url} | {title} |")
    lines.append("")
    content = "\n".join(lines)
    if not dry_run:
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            f.write(content)
    return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if shutil.which("pandoc") is None:
        print("ERROR: pandoc not found on PATH. This skill needs it for HTML->Markdown "
              "conversion. On Sol it's normally at /usr/bin/pandoc already; if it's "
              "missing, install it or module-load it before rerunning.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    atx_flag = _pandoc_atx_flag()

    print(f"Fetching sitemap from {SITEMAP_URL} ...")
    pairs = discover_urls()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} doc pages to check.\n")

    stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "failed": 0}
    changed_files = []
    failures = []
    index_entries = []
    seen_filenames = set()

    for i, (path, url) in enumerate(pairs, 1):
        filename = path_to_filename(path)
        seen_filenames.add(filename)
        if args.verbose:
            print(f"[{i}/{len(pairs)}] {url}", file=sys.stderr)
        try:
            result, err = process_page(path, url, atx_flag, verbose=args.verbose)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            result, err = None, str(e)

        if result is None:
            stats["failed"] += 1
            failures.append((url, err))
            existing = load_existing(filename)
            if existing:
                # Keep serving the last known-good title/url in the index
                # rather than dropping the row just because this run failed.
                m = re.search(r"<!-- source: (.*?) -->\n# (.*)", existing)
                index_entries.append((filename, m.group(1) if m else url, m.group(2) if m else filename))
            continue

        content, title = result
        existing = load_existing(filename)
        index_entries.append((filename, url, title))
        if existing is None:
            stats["added"] += 1
            changed_files.append(("added", filename))
            write_file(filename, content, args.dry_run)
        elif existing != content:
            stats["updated"] += 1
            changed_files.append(("updated", filename))
            write_file(filename, content, args.dry_run)
        else:
            stats["unchanged"] += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    # Stale-file cleanup: local .md files whose page is no longer in the
    # (filtered) sitemap at all. Skipped entirely on --limit runs, since a
    # partial crawl would otherwise look like most of the site disappeared.
    if not args.limit:
        existing_files = {
            f for f in os.listdir(OUTPUT_DIR)
            if f.endswith(".md") and f not in PROTECTED_FILES
        }
        stale = sorted(existing_files - seen_filenames)
        for filename in stale:
            stats["removed"] += 1
            changed_files.append(("removed", filename))
            if not args.dry_run:
                os.remove(os.path.join(OUTPUT_DIR, filename))

    index_entries.sort(key=lambda e: e[0])
    run_date = time.strftime("%Y-%m-%d")
    rebuild_index(index_entries, run_date, stats, args.dry_run)

    print("\n--- Summary ---")
    print(json.dumps(stats, indent=2))
    if changed_files:
        print("\nChanged files:")
        for action, filename in changed_files:
            print(f"  {action:8s} {filename}")
    if failures:
        print("\nFetch failures (previous cached copy kept, if any):")
        for url, err in failures:
            print(f"  {url}: {err}")
    if args.dry_run:
        print("\n(--dry-run: no files were actually written)")


if __name__ == "__main__":
    main()
