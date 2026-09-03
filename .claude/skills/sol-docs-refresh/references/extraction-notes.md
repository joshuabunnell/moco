# Extraction pipeline notes

How `scripts/refresh.py` turns a docs.rc.asu.edu page into a clean local
`.md` file, and the bugs already found while building it (2026-08-19) —
read this before re-deriving a fix if something breaks again.

## Pipeline order (per page)

1. `fetch()` the page's trailing-slash URL (see "308 redirects" below).
2. `extract_content_html()` — pull just the content wrapper's inner HTML
   out of the full page, using `ClassExtractor` (a depth-tracking
   `HTMLParser`), trying `theme-doc-markdown` then `generatedIndexPage`.
3. `html_to_markdown()` — strip breadcrumb nav and pagination nav (regex on
   the HTML fragment), run `rewrite_tabs()` to label Docusaurus tab panels,
   drop `<img>` tags, then pipe through `pandoc -f html -t markdown-simple_tables+pipe_tables <atx flag> --wrap=none`.
4. `clean_markdown()` — de-zero-width-space, then protect fenced code
   blocks from the rest of the cleanup (see "Code block corruption" below),
   then per prose segment: collapse admonition boxes to `**Label:**`, strip
   anchor-permalink cruft, re-attach card-link hrefs to their headings,
   strip leftover pandoc attribute spans and fence markers, strip orphaned
   `<div>`/`<!-- -->` HTML, un-escape pandoc's backslash-escaped punctuation,
   demote heading levels by one (page's own `# H1` becomes our `## H2`,
   since the file always starts with `# {title} | ASU RC Docs`).

## Bugs found while building this (all fixed, but the *shape* of bug is
worth knowing if the site's template changes)

**Sentinel bug in `ClassExtractor` (over-capture).** The extractor used
`self.capture_at_depth = "done"` as a sentinel meaning "stop capturing," but
every subsequent check was `if self.capture_at_depth is not None`, which is
true for the string `"done"` too — so it kept capturing everything
(pagination, footer, the works) all the way to the end of the document.
Fixed by checking `isinstance(self.capture_at_depth, int)` everywhere
instead of `is not None`. If output for some page looks like it's grabbing
way more than the visible content div, this is the first thing to check —
the same footgun (truthy-non-None sentinel used where an int-only check
was needed) is easy to reintroduce if this class gets extended.

**`${VAR}` and `awk '{...}'` corruption inside code blocks.** The cleanup
regex that strips pandoc's own `{.class}` attribute spans (`ATTR_SPAN_RE`)
was, before the fix, applied to the *entire* page text — including inside
fenced code blocks. Shell parameter expansion (`${SLURM_JOB_ID}`) and
awk/sed program blocks (`'{print $1}'`) are syntactically indistinguishable
from pandoc's attribute-span shape, so real commands on the software pages
(abaqus, gromacs, etc.) were getting silently mangled — `${HOSTS_FILE}`
became just `$`. Fixed by splitting on fenced code blocks first
(`CODE_FENCE_SPLIT_RE`) and running the attribute/escape/heading regexes
only on the non-code segments (`_clean_prose`), leaving code content
completely untouched. **Any new cleanup regex added to this pipeline must
go inside `_clean_prose`, never applied to the whole document — otherwise
it will eventually corrupt example code on some software page.**

**Indented code fences inside numbered lists weren't recognized as code at
all.** Both `CODE_FENCE_SPLIT_RE` (protects code from cleanup) and
`BARE_FENCE_RE` (strips leftover pandoc div-fence markers) originally
anchored on `^```` / `^:::` with no leading whitespace allowed. Docusaurus
numbered-step pages (vscode.md's tunnel walkthrough, for one) put code
samples inside list items, which pandoc indents by 4 spaces — so those
fences were invisible to both regexes: not protected from cleanup (latent
corruption risk, just didn't happen to hit anything with `{}` in the
sampled pages) and left visible stray `:::` lines in the output. Fixed by
allowing `[ \t]*` before the fence markers in both regexes (and in
`CODE_FENCE_OPEN_RE`, which simplifies the language tag on the opening
fence).

**Docusaurus tabs flatten with no labels.** A `:::tabs` block in the source
MDX renders as a `role="tablist"` `<ul>` of labels followed by one
`role="tabpanel"` `<div>` per label (all present in the DOM; CSS `hidden`
picks which one shows). Converted to markdown, this reads as one undifferentiated
block of content with nothing marking where each label's content starts —
harmless for tabs that are just alternative phrasing, actively misleading
for tabs that disambiguate content (per-OS install commands, per-role
instructions, per-partition `sbatch`/`salloc` variants — this site uses
tabs for exactly this a lot, e.g. `/partitions-and-qos`). Fixed with
`TabRewriter`, a full pass-through `HTMLParser` rewrite that collects each
tablist's label text and injects a synthetic `<h3>{label}</h3>` as the
first child of the next tabpanel, consuming labels in document order.
Multiple tab groups per page work as long as each group's panels appear
before the next group's tablist — true for every Docusaurus-rendered page.

**Block-wrapping card links produce dangling `[](url)`.** Category-index
pages (`/accounts`, `/connecting`, `/computing`, ...) wrap each card in a
single `<a>` containing both its heading and description. Pandoc can't
express a link wrapping block content as inline markdown, so it emits the
bare href as an empty `[](url)` line immediately before the heading it
belonged to, rather than making the heading itself a link. Fixed with
`CARD_LINK_RE`, which re-attaches the href to the following heading:
`### [Title](/url)` instead of `[](/url)` + `### Title` as two disconnected
lines. Note the href and heading aren't always adjacent — pandoc sometimes
leaves the card wrapper's own attribute span (`{.card ...}`) attached to
the closing paren first, so the regex has to tolerate an optional
`{...}` between the link and the newline before the heading.

## Non-bugs / accepted limitations

- **308 redirects.** The site 308-redirects any non-trailing-slash route to
  its trailing-slash form (Cloudflare-fronted). `urllib` on this system's
  Python (3.6.8) doesn't auto-follow 308s (that landed in a newer stdlib).
  Worked around by always fetching `url + "/"` directly rather than relying
  on redirect-following — not worth a general redirect handler for a single
  known, permanent redirect shape.
- **Tab list labels also render as a redundant bullet list.** The `<li>`
  label text still emits normally as part of the tab list markup (e.g. "-
  Select a role\n- I am a Student\n...") *in addition to* being used for the
  injected `### I am a Student` headings. Mildly repetitive but not
  incorrect, and removing it cleanly would mean suppressing the tablist
  `<ul>`'s normal rendering entirely — more surface area for something to
  break than the redundancy is worth.
- **Old system pandoc (2.0.6).** `--markdown-headings=atx` was renamed from
  `--atx-headers` in pandoc 2.11.4; Sol's system pandoc predates that.
  `_pandoc_atx_flag()` detects which flag `pandoc --help` supports at
  runtime, so this keeps working whether pandoc gets upgraded or the script
  runs somewhere newer via `module load`.
- **Python 3.6.8 compatibility.** No `subprocess.run(capture_output=...)`
  (3.7+) — uses `stdout=PIPE, stderr=PIPE` instead. No other 3.7+-only
  syntax was knowingly used, but if this ever runs on a different Sol login
  node or gets copied elsewhere, re-check for f-string edge cases and
  dict-ordering assumptions (both fine on 3.6.8's CPython, but not
  guaranteed by the language spec until 3.7).
