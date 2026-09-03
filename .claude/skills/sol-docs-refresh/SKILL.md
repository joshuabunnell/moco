---
name: sol-docs-refresh
description: Refreshes the local ASU Research Computing docs mirror at ~/sol-docs/ from the live docs.rc.asu.edu site — re-crawls the whole site via its sitemap, re-scrapes every page to clean markdown, and rewrites only what actually changed (plus removes pages deleted upstream). Use this whenever the user says anything like "make sure sol docs are updated," "refresh the sol docs," "check if the ASU RC docs changed," "sol-docs is stale," "pull the latest HPC docs," or "update ~/sol-docs" — even if they don't name the skill directly. Also reach for this proactively if you're about to answer a question from ~/sol-docs and its staleness matters (e.g. anything about accounts, Voyager, partitions/QoS, or other content that changes over time) — offer to refresh first rather than silently trusting a snapshot that could be out of date.
---

# sol-docs-refresh

## Why this exists

`~/sol-docs/` is a local markdown mirror of `docs.rc.asu.edu` (ASU Research
Computing's documentation site for the Sol and Phoenix supercomputers), kept
at user root because it's environment reference, not project code — see any
project's `CLAUDE.md` for how it's used day to day. It started as a one-time
manual crawl and had drifted out of sync with the live site (missing the
whole Voyager self-service rollout, ~90 pages that didn't exist yet, etc.).
This skill is the maintenance path: run it any time you want the cache
current again, instead of re-deriving pages by hand.

**Confirmed source of truth:** `docs.rc.asu.edu` is the correct domain — ASU
RC's own account-management emails link directly to pages under this domain
(e.g. `docs.rc.asu.edu/voyager-request-account`), which is what prompted this
skill to begin with. If a future email or doc ever points somewhere else,
that's a real signal the docs moved and this skill's `BASE` constant needs
updating — don't assume it silently.

## What it does

1. Fetches `https://docs.rc.asu.edu/sitemap.xml` to discover every real page.
   There's no cheaper way to know what changed upstream — the sitemap
   carries no `lastmod` dates — so this always re-fetches everything and
   diffs locally rather than trying to guess what's new.
2. Excludes the blog-style sections (`/changelog`, `/events`, `/news`,
   `/search`, tag/author index pages) — churn, not reference material — but
   otherwise scrapes the **whole site**: general HPC/account topics, every
   per-application page (abaqus, gromacs, vasp, alphafold, ...), cloud/VM
   topics, the AI/LLM API section, and the tutorials. This was a deliberate
   scope call (confirmed with the user 2026-08-19) over keeping the old
   curated ~39-page subset — simpler to maintain with no per-page judgment
   calls, and matches "all docs for this supercomputer" literally.
3. Converts each page to clean markdown and writes it to `~/sol-docs/<slug>.md`
   **only if the content actually changed** — untouched pages don't show up
   in git history or as noise to re-read.
4. Deletes local `.md` files whose page is no longer in the sitemap at all
   (confirmed absent, not just a fetch hiccup — see Safety below). Also
   confirmed with the user: stale pages are removed automatically, not just
   flagged.
5. Rewrites `INDEX.md` from the current file set every run.

Two files are permanently out of scope and never touched:
- `policies.md` — lives on a *different* domain (`cores.research.asu.edu`),
  not part of this site's sitemap or template structure. Update by hand if
  ASU RC ever changes their acceptable-use policy.
- `home.md` — the landing page is a bespoke hero/marketing layout with no
  extractable content div the way every other page has one. It's a
  hand-curated summary; leave it alone or rewrite it by hand if it goes
  stale, don't expect this script to touch it.

## Running it

```bash
python3 ~/moco/.claude/skills/sol-docs-refresh/scripts/refresh.py
```

Useful flags:
- `--dry-run` — report what would change without writing anything. **Do this
  first** if it's been a while since the last run (the site adds pages
  fairly often) or if you're not sure what to expect — skim the summary
  before committing to a real run that deletes things.
- `--limit N` — only process the first N sitemap pages. For quick sanity
  checks; note it also skips the stale-file cleanup pass entirely (a partial
  crawl would otherwise look like most of the site vanished).
- `--verbose` — print each page as it's fetched, with which content wrapper
  matched. Useful if something's failing and you want to see where.

A full run is ~130 pages at ~0.3s delay between requests plus fetch/pandoc
time — expect roughly 2-4 minutes. It's pure Python stdlib + the system
`pandoc` binary (already at `/usr/bin/pandoc` on Sol as of this writing); no
pip installs, no API keys, nothing else to set up.

**After running**, tell the user what changed — don't just say "done."
Summarize from the script's own summary block: counts, and the standout
files (new pages worth knowing about, anything with real content changes
like the Voyager rollout previously). If a page's *meaning* changed (not
just formatting drift), call that out specifically rather than burying it
in a file list — that's usually what the user actually cares about.

## Safety notes

- **Fetch failures never delete anything.** A page erroring out (timeout,
  transient 5xx, etc.) keeps its previous cached copy and gets reported
  separately as a failure, not folded into "removed." Only a page's
  confirmed *absence* from the current sitemap triggers deletion.
- The script writes files as it goes but only rewrites `INDEX.md` once at
  the end, and only for a real (non-`--dry-run`) run — a run that dies
  partway through leaves already-updated pages in place and the old INDEX.md
  a little stale until the next successful run, rather than a half-written
  index.
- If you want to review before committing to deletions, use `--dry-run`
  first and read the "removed" lines in its summary before running for
  real.

## If it starts failing broadly

Check `references/extraction-notes.md` for how the HTML→markdown pipeline
works and the specific bugs already found and fixed while building this
(over-capture past the intended content div, `${VAR}`/`awk '{...}'`
corruption inside code blocks, indented code fences inside numbered lists,
Docusaurus tabs flattening with no labels, etc.) — useful context before
re-deriving a fix from scratch if the site's template changes again.

A single page failing with "no known content wrapper found" usually means
Docusaurus shipped a new page template; add its wrapper class to
`CONTENT_CLASS_CANDIDATES` in `scripts/refresh.py`. Many pages failing at
once more likely means the site's front end changed structurally, or
`pandoc`/Python moved — check `pandoc --version` still exists and the
`_pandoc_atx_flag()` detection in the script still finds a working flag.
