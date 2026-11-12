"""CI only: screenshot every EchoMap tab after its data has loaded (Selenium + Edge on the Windows runner).
Usage: python packaging/ui_screens.py <app url> <out dir> [examples.json]
"""
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait

url, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
opt = Options()
opt.add_argument("--headless=new")
opt.add_argument("--window-size=1480,1000")
opt.add_argument("--hide-scrollbars")
d = webdriver.Edge(options=opt)
d.get(url)
wait = WebDriverWait(d, 90)
wait.until(lambda drv: drv.find_element(By.ID, "invCount").text not in ("", "0"))
time.sleep(3)


def tab(name, settle, ready=None):
    d.find_element(By.CSS_SELECTOR, f'.tab[data-tab="{name}"]').click()
    if ready:
        try:
            WebDriverWait(d, 30).until(ready)
        except Exception:  # noqa: BLE001
            pass
    time.sleep(settle)


tab("analysis", 4)
d.save_screenshot(str(out / "1-analysis.png"))
tab("inventory", 2)
d.execute_script("document.querySelectorAll('#inventory details.det')[0]?.setAttribute('open','')")
time.sleep(1)
d.save_screenshot(str(out / "2-inventory.png"))
tab("geo", 6, lambda drv: drv.find_element(By.ID, "clueList").text.strip() not in ("", "No clues yet."))
d.save_screenshot(str(out / "3-geo-clue-map.png"))
d.find_element(By.CSS_SELECTOR, '.geov[data-v="globe"]').click()
time.sleep(4)
d.save_screenshot(str(out / "3b-geo-globe.png"))
d.find_element(By.CSS_SELECTOR, '.geov[data-v="clue"]').click()
tab("intel", 3)
d.save_screenshot(str(out / "4-intelligence.png"))
tab("report", 6)
d.save_screenshot(str(out / "5-report.png"))
# v1.0.3: worked examples rendered by the real UI (clue sets from tools/example_geo.py, injected into the Geo view)
if len(sys.argv) > 3:
    import json
    ex = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    tab("geo", 2)
    for name in ("prague-tram", "tram-announcement", "melbourne-tram"):
        if name not in ex:
            continue
        d.execute_script("Geo.render(arguments[0]);", ex[name])
        time.sleep(4)   # auto-zoom to the likely places
        d.save_screenshot(str(out / f"6-example-{name}-zoomed.png"))
        d.find_element(By.ID, "clReset").click()
        time.sleep(2)
        d.save_screenshot(str(out / f"6-example-{name}-world.png"))
d.quit()
print("screenshots:", sorted(p.name for p in out.glob("*.png")))
