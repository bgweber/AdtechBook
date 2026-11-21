"""Make the EPUB a native ebook, not a straight conversion of the print book.

Quarto runs this after every render (project: post-render in _quarto.yml).
It only touches the EPUB, so the web and print editions don't change. It:

- adds a divider page for each part (Quarto drops parts from EPUB output),
- rebuilds the table of contents as parts > chapters, without every section,
- adds a copyright page for the ebook (no paperback ISBN),
- removes the raw build date from the title page,
- adds landmarks so readers open the book at the start of the content,
- adds the cover (metadata and a plain cover page).

Run it by hand with: python scripts/polish_epub.py _book/ml-in-mobile-adtech.epub
"""

import html
import os
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
YEAR = date.today().year

COPYRIGHT_BODY = f"""\
<p class="copyright-title"><em>Machine Learning in Mobile AdTech:
From Foundations to the Frontier of Demand Side Platforms</em></p>
<p>Copyright &#169; {YEAR} Ben G Weber.</p>
<p>The text and figures of this book are licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License
(<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>),
except the cover and the author photo. The code, including the code samples
in this book, is licensed under the MIT License. The code and the bid
simulator are available at
<a href="https://github.com/bgweber/AdtechBook">github.com/bgweber/AdtechBook</a>.</p>
<p>A free online edition is available at
<a href="https://bgweber.github.io/AdtechBook/">bgweber.github.io/AdtechBook</a>.</p>
<p>While every precaution has been taken in preparing this book, the author
assumes no responsibility for errors or omissions, or for damages resulting
from the use of the information it contains.</p>
<p>Free ebook edition, {YEAR}.</p>
"""


def page(title, body, body_type, extra_head=""):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="../styles/stylesheet1.css" />{extra_head}
</head>
<body epub:type="{body_type}">
{body}
</body>
</html>
"""


def read_parts():
    """Return [(part title, id of the part's first chapter)] from _quarto.yml."""
    lines = (ROOT / "_quarto.yml").read_text(encoding="utf-8").splitlines()
    parts = []
    for i, line in enumerate(lines):
        m = re.match(r"\s*- part:\s*(\S+)", line)
        if not m:
            continue
        part_file = ROOT / m.group(1)
        heading = next(l for l in part_file.read_text(encoding="utf-8").splitlines()
                       if l.startswith("# "))
        title = re.sub(r"\s*\{.*\}\s*$", "", heading[2:]).strip()
        first = next(re.match(r"\s*- (\S+\.qmd)", l).group(1)
                     for l in lines[i + 1:] if re.match(r"\s*- \S+\.qmd", l))
        first_heading = next(l for l in (ROOT / first).read_text(encoding="utf-8").splitlines()
                             if l.startswith("# "))
        sec = re.search(r"\{#([^\s}]+)", first_heading)
        parts.append((title, sec.group(1) if sec else None))
    return parts


def h1_text(xhtml):
    m = re.search(r"<h1[^>]*>(.*?)</h1>", xhtml, re.S)
    if not m:
        return None
    inner = m.group(1)
    num = re.search(r'<span class="header-section-number">([^<]*)</span>', inner)
    text = html.unescape(re.sub(r"<[^>]+>", "", inner)).strip()
    if num:
        text = text[len(num.group(1)):].strip()
        return f"{num.group(1)}. {text}"
    return text


def polish(epub_path, cover_page=True):
    epub_path = Path(epub_path)
    with zipfile.ZipFile(epub_path) as z:
        files = {n: z.read(n) for n in z.namelist()}
    if any(n.endswith("text/copyright.xhtml") for n in files):
        return  # already polished
    opf_name = next(n for n in files if n.endswith(".opf"))
    base = opf_name.rsplit("/", 1)[0] + "/"
    opf = files[opf_name].decode("utf-8")

    spine = re.findall(r'<itemref idref="([^"]+)"', opf)
    href = dict(re.findall(r'<item id="([^"]+)" href="([^"]+)"', opf))
    texts = {i: files[base + href[i]].decode("utf-8") for i in spine
             if href[i].startswith("text/")}

    # Quarto copies fig-alt onto the figure's wrapper <div> as well as the
    # <img>. Only the <img> may carry alt text, so drop it from the <div>.
    for i in list(texts):
        fixed = re.sub(r'(<div\b[^>]*?)\s+alt="[^"]*"', r"\1", texts[i])
        if fixed != texts[i]:
            texts[i] = fixed
            files[base + href[i]] = fixed.encode("utf-8")

    # Title page: drop the raw build date.
    tp = base + "text/title_page.xhtml"
    files[tp] = re.sub(r'\s*<p class="date">.*?</p>', "",
                       files[tp].decode("utf-8")).encode("utf-8")

    # Part divider pages, before each part's first chapter.
    new_items, before = [], {}
    for n, (title, sec) in enumerate(read_parts()):
        target = next((i for i, t in texts.items() if sec and f'id="{sec}"' in t), None)
        if target is None:
            continue
        pid = f"part{n + 1:02d}_xhtml"
        name = f"text/part{n + 1:02d}.xhtml"
        body = (f'<section epub:type="part" class="part-page" id="part-{n + 1}">\n'
                f'  <p class="part-label">Part {ROMAN[n]}</p>\n'
                f'  <h1 class="part-title">{html.escape(title)}</h1>\n</section>')
        files[base + name] = page(f"Part {ROMAN[n]}: {title}", body,
                                  "bodymatter").encode("utf-8")
        new_items.append(f'<item id="{pid}" href="{name}" media-type="application/xhtml+xml" />')
        before[target] = (pid, f"Part {ROMAN[n]}: {title}", name)

    # Copyright page, after the title page.
    files[base + "text/copyright.xhtml"] = page(
        "Copyright",
        f'<section epub:type="copyright-page" class="copyright-page">\n{COPYRIGHT_BODY}</section>',
        "frontmatter").encode("utf-8")
    new_items.append('<item id="copyright_xhtml" href="text/copyright.xhtml" '
                     'media-type="application/xhtml+xml" />')

    # Manifest and spine.
    opf = opf.replace("</manifest>", "    " + "\n    ".join(new_items) + "\n  </manifest>")
    new_spine = []
    for i in spine:
        if i in before:
            new_spine.append(before[i][0])
        new_spine.append(i)
        if i == "title_page_xhtml":
            new_spine.append("copyright_xhtml")
    opf = re.sub(r"<spine([^>]*)>.*?</spine>",
                 lambda m: f"<spine{m.group(1)}>\n" + "".join(
                     f'    <itemref idref="{i}" />\n' for i in new_spine) + "  </spine>",
                 opf, flags=re.S)

    # Table of contents: front matter, parts > chapters, appendices.
    toc = []  # (label, href, children)
    current = None
    for i in spine:
        if i not in texts or i == "title_page_xhtml":
            continue
        label = h1_text(texts[i])
        if not label:
            continue
        if i in before:
            current = (before[i][1], before[i][2], [])
            toc.append(current)
        if label.startswith("Appendix"):
            if not current or current[0] != "Appendices":
                current = ("Appendices", href[i], [])
                toc.append(current)
            label = label.replace("Appendix ", "", 1).replace(" — ", ". ", 1)
        if current:
            current[2].append((label, href[i], []))
        else:
            toc.append((label, href[i], []))

    def nav_ol(entries, indent):
        pad = " " * indent
        out = [f"{pad}<ol>"]
        for label, h, kids in entries:
            out.append(f'{pad}  <li><a href="{h}">{html.escape(label)}</a>')
            if kids:
                out.append(nav_ol(kids, indent + 4))
            out.append(f"{pad}  </li>")
        out.append(f"{pad}</ol>")
        return "\n".join(out)

    start = next(h for label, h, _ in toc if label == "Preface")
    nav_name = base + next(h for i, h in href.items() if i == "nav")
    nav = files[nav_name].decode("utf-8")
    nav = re.sub(r'(<nav epub:type="toc"[^>]*>).*?</nav>',
                 lambda m: m.group(1) + "\n  <h1>Contents</h1>\n" + nav_ol(toc, 2) + "\n</nav>",
                 nav, flags=re.S)
    landmarks = [("titlepage", "text/title_page.xhtml", "Title Page"),
                 ("copyright-page", "text/copyright.xhtml", "Copyright"),
                 ("toc", "#toc", "Table of Contents"),
                 ("bodymatter", start, "Start of Content")]
    nav = re.sub(r'(<nav epub:type="landmarks"[^>]*>).*?</nav>',
                 lambda m: m.group(1) + "\n  <ol>\n" + "".join(
                     f'    <li><a href="{h}" epub:type="{t}">{l}</a></li>\n'
                     for t, h, l in landmarks) + "  </ol>\n</nav>",
                 nav, flags=re.S)
    files[nav_name] = nav.encode("utf-8")
    opf = re.sub(r"<guide>.*?</guide>",
                 '<guide>\n    <reference type="toc" title="Table of Contents" href="nav.xhtml" />\n'
                 f'    <reference type="text" title="Start of Content" href="{start}" />\n  </guide>',
                 opf, flags=re.S)
    files[opf_name] = opf.encode("utf-8")

    # NCX, for older Kindles.
    ncx_name = next(n for n in files if n.endswith(".ncx"))
    ncx = files[ncx_name].decode("utf-8")
    counter = [0]

    def nav_points(entries, indent):
        out = []
        for label, h, kids in entries:
            counter[0] += 1
            pad = " " * indent
            out.append(f'{pad}<navPoint id="navPoint-{counter[0]}" playOrder="{counter[0]}">\n'
                       f"{pad}  <navLabel><text>{html.escape(label)}</text></navLabel>\n"
                       f'{pad}  <content src="{h}" />')
            if kids:
                out.append(nav_points(kids, indent + 2))
            out.append(f"{pad}</navPoint>")
        return "\n".join(out)

    ncx = re.sub(r"<navMap>.*?</navMap>",
                 lambda m: "<navMap>\n" + nav_points(toc, 4) + "\n  </navMap>", ncx, flags=re.S)
    ncx = re.sub(r'<meta name="dtb:depth" content="\d+" />',
                 '<meta name="dtb:depth" content="2" />', ncx)
    files[ncx_name] = ncx.encode("utf-8")

    # Cover. Quarto's epub-cover-image wraps the image in an SVG page, which
    # made Kindle Previewer's conversion fail, so add it the plain way: the
    # image marked as the cover in the metadata (EPUB 3 property and the
    # older meta tag Kindle reads), and a simple first page with an <img>.
    cover_src = ROOT / "images" / "cover.jpg"
    if cover_src.exists():
        files[base + "media/cover.jpg"] = cover_src.read_bytes()
        opf = files[opf_name].decode("utf-8")
        items = ['<item id="cover-image" href="media/cover.jpg" media-type="image/jpeg" '
                 'properties="cover-image" />']
        if cover_page:
            files[base + "text/cover.xhtml"] = page(
                "Cover",
                '<div class="cover-page"><img src="../media/cover.jpg" '
                'alt="Cover of Machine Learning in Mobile AdTech by Ben G Weber" /></div>',
                "cover").encode("utf-8")
            items.append('<item id="cover_xhtml" href="text/cover.xhtml" '
                         'media-type="application/xhtml+xml" />')
            opf = re.sub(r"(<spine[^>]*>\s*)", r'\1<itemref idref="cover_xhtml" />\n    ',
                         opf, count=1)
        opf = opf.replace("</manifest>", "  " + "\n    ".join(items) + "\n  </manifest>")
        opf = opf.replace("</metadata>", '  <meta name="cover" content="cover-image" />\n  </metadata>')
        if cover_page:
            opf = opf.replace("<guide>", '<guide>\n    <reference type="cover" title="Cover" href="text/cover.xhtml" />', 1)
        files[opf_name] = opf.encode("utf-8")

    # Write back: mimetype first and uncompressed, as EPUB requires.
    tmp = epub_path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"),
                   compress_type=zipfile.ZIP_STORED)
        for n, data in files.items():
            z.writestr(n, data, compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, epub_path)
    print(f"polish_epub: {len(before)} part pages, copyright page, "
          f"{sum(len(k) or 1 for _, _, k in toc)} contents entries -> {epub_path}")


def main():
    paths = sys.argv[1:]
    if not paths:
        out_files = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "")
        paths = [p for p in out_files.splitlines() if p.endswith(".epub")]
    # POLISH_EPUB_COVER_PAGE=0 keeps the cover as metadata only (the library
    # thumbnail), with no cover page in the book.
    cover_page = os.environ.get("POLISH_EPUB_COVER_PAGE", "1") != "0"
    for p in paths:
        if Path(p).exists():
            polish(p, cover_page=cover_page)


if __name__ == "__main__":
    main()
