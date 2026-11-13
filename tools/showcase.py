"""Screenshots of the real UI for an analysed case (CI only; Playwright + Chromium on the runner).

shoot(case, out_dir, name) serves the case through EchoMap's own local server and captures: the Findings panel,
the Analysis tab, the Detection inventory, the clue map (Geographic tab) and, when present, the first animal /
vehicle detection's "where in the world" map. Used by tools/soundscape_bench.py and the showcase workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap import server  # noqa: E402


def shoot(case, out_dir: Path, name: str) -> list[str]:
    from playwright.sync_api import sync_playwright
    out_dir.mkdir(parents=True, exist_ok=True)
    st = server.AppState(case.settings)
    st.case = case
    port = server.start(st)
    url = f"http://127.0.0.1:{port}/"
    shots = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1480, "height": 1000}, device_scale_factor=1)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        pg.wait_for_timeout(1500)
        pg.evaluate("loadResult()")
        pg.wait_for_timeout(2500)

        def snap(fname, selector=None, full=False):
            f = out_dir / f"{name}_{fname}.png"
            if selector:
                el = pg.query_selector(selector)
                if el and el.is_visible():
                    el.screenshot(path=str(f))
                else:
                    return
            else:
                pg.screenshot(path=str(f), full_page=full)
            shots.append(str(f))
        snap("1_findings", "#findings")
        snap("2_analysis")
        pg.click('.tab[data-tab="inventory"]')
        pg.wait_for_timeout(800)
        pg.evaluate("document.querySelectorAll('#inventory details.det').forEach((d, i) => { if (i < 4) d.open = true; })")
        snap("3_inventory")
        pg.click('.tab[data-tab="geo"]')
        pg.wait_for_timeout(3000)
        snap("4_map")
        link = pg.query_selector("#findings .fmap")
        if link:
            pg.click('.tab[data-tab="analysis"]')
            pg.wait_for_timeout(500)
            pg.click("#findings .fmap")
            pg.wait_for_timeout(3000)
            snap("5_where_found")
        if errs:
            (out_dir / f"{name}_js_errors.txt").write_text("\n".join(errs), encoding="utf-8")
        b.close()
    return shots
