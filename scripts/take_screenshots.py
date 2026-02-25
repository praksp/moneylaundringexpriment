"""
scripts/take_screenshots.py
Captures screenshots of every UI page for the AML application.
Run from the project root with the API (port 8001) and frontend (port 5174) running:
    python scripts/take_screenshots.py
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, Page

BASE = "http://localhost:5174"
OUT  = Path("screenshots")
OUT.mkdir(exist_ok=True)

VIEWPORT = {"width": 1440, "height": 900}


def wait_network_idle(page: Page, timeout: int = 8001):
    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception:
        pass


def login(page: Page, username="admin", password="password"):
    page.goto(f"{BASE}/login", wait_until="domcontentloaded")
    wait_network_idle(page)
    time.sleep(1)
    page.fill("input[type='text'], input[placeholder*='sername'], input[name='username']", username)
    page.fill("input[type='password']", password)
    page.screenshot(path=str(OUT / "login.png"), full_page=False)
    print("  ✓ login.png (pre-submit)")
    page.click("button[type='submit']")
    wait_network_idle(page)
    time.sleep(2)


def ss(page: Page, filename: str, scroll_to_bottom: bool = False, extra_wait: float = 0.5):
    if scroll_to_bottom:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.5)
    else:
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.3)
    time.sleep(extra_wait)
    page.screenshot(path=str(OUT / filename), full_page=False)
    print(f"  ✓ {filename}")


def run():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(viewport=VIEWPORT)
        page = ctx.new_page()

        # ── 1. Login ──────────────────────────────────────────────────────────
        print("1. Login page...")
        login(page)

        # ── 2. Dashboard ──────────────────────────────────────────────────────
        print("2. Dashboard...")
        page.goto(f"{BASE}/", wait_until="domcontentloaded")
        wait_network_idle(page, 10000)
        time.sleep(3)
        ss(page, "dashboard.png")

        # ── 3. Submit Transaction ─────────────────────────────────────────────
        print("3. Submit Transaction...")
        page.goto(f"{BASE}/transactions/new", wait_until="domcontentloaded")
        wait_network_idle(page, 8001)
        time.sleep(2)
        # Fill in a sample transaction
        try:
            # Amount field
            amt_el = page.locator("input[placeholder*='mount'], input[name*='mount']").first
            amt_el.fill("9500")
            time.sleep(0.3)
            # Try to select currency USD
            cur_el = page.locator("select").first
            cur_el.select_option("USD")
            time.sleep(0.3)
        except Exception:
            pass
        ss(page, "submit-transaction.png")

        # ── 4. Transaction result / recent transactions list ─────────────────
        print("4. Transactions list...")
        page.goto(f"{BASE}/transactions", wait_until="domcontentloaded")
        wait_network_idle(page, 10000)
        time.sleep(3)
        ss(page, "transaction-list.png")

        # Click first transaction row if available
        try:
            first_row = page.locator("tbody tr, [data-testid='txn-row']").first
            first_row.click(timeout=3000)
            wait_network_idle(page, 6000)
            time.sleep(2)
            ss(page, "transaction-detail.png")
            # Go back
            page.go_back()
            time.sleep(1)
        except Exception as e:
            print(f"    (no detail: {e})")

        # ── 5. Customer Profile ───────────────────────────────────────────────
        print("5. Customer Profile...")
        page.goto(f"{BASE}/customers", wait_until="domcontentloaded")
        wait_network_idle(page, 12000)
        time.sleep(4)
        ss(page, "customer-profile.png")

        # Click first customer
        try:
            first_customer = page.locator("tbody tr, [data-customer], button[data-id]").first
            first_customer.click(timeout=3000)
            wait_network_idle(page, 8001)
            time.sleep(3)
            ss(page, "customer-detail.png")
        except Exception as e:
            print(f"    (no customer detail: {e})")

        # ── 6. Model Monitor ─────────────────────────────────────────────────
        print("6. Model Monitor...")
        page.goto(f"{BASE}/monitoring", wait_until="domcontentloaded")
        wait_network_idle(page, 10000)
        time.sleep(3)
        ss(page, "model-monitor.png")

        # ── 7. Feature Store ─────────────────────────────────────────────────
        print("7. Feature Store...")
        page.goto(f"{BASE}/feature-store", wait_until="domcontentloaded")
        wait_network_idle(page, 10000)
        time.sleep(3)
        ss(page, "feature-store.png")

        # Expand first row if there's a details button
        try:
            expand_btn = page.locator("button[aria-expanded], button:has-text('Details'), tr button").first
            expand_btn.click(timeout=3000)
            time.sleep(1.5)
            ss(page, "feature-store-expanded.png")
        except Exception:
            pass

        # ── 8. GraphSAGE Mule Detection ──────────────────────────────────────
        print("8. GraphSAGE Detection...")
        page.goto(f"{BASE}/graphsage", wait_until="domcontentloaded")
        wait_network_idle(page, 12000)
        time.sleep(4)
        ss(page, "graphsage-detection.png")

        # Click first suspect row to open drawer
        try:
            suspect_row = page.locator("tbody tr").first
            suspect_row.click(timeout=4000)
            wait_network_idle(page, 6000)
            time.sleep(3)
            ss(page, "graphsage-drawer.png")
            # Switch to transactions tab if visible
            try:
                txn_tab = page.locator("button:has-text('Fraud Txns')").first
                txn_tab.click(timeout=2000)
                time.sleep(1.5)
                ss(page, "graphsage-drawer-txns.png")
            except Exception:
                pass
            # Close drawer
            try:
                page.keyboard.press("Escape")
                time.sleep(1)
            except Exception:
                pass
        except Exception as e:
            print(f"    (no drawer: {e})")

        # How It Works tab
        try:
            howto_tab = page.locator("button:has-text('How It Works')").first
            howto_tab.click(timeout=3000)
            wait_network_idle(page, 4000)
            time.sleep(2)
            ss(page, "graphsage-howto.png")
        except Exception as e:
            print(f"    (no howto: {e})")

        # ── 9. KNN Anomaly Detection ─────────────────────────────────────────
        print("9. KNN Anomaly Detection...")
        page.goto(f"{BASE}/anomaly", wait_until="domcontentloaded")
        wait_network_idle(page, 10000)
        time.sleep(3)
        ss(page, "knn-anomaly.png")

        # ── 10. Dashboard (scroll to show world map) ─────────────────────────
        print("10. Dashboard world-map view...")
        page.goto(f"{BASE}/", wait_until="domcontentloaded")
        wait_network_idle(page, 10000)
        time.sleep(3)
        # Scroll to show the map area
        page.evaluate("window.scrollTo(0, 400)")
        time.sleep(2)
        ss(page, "dashboard-map.png")

        browser.close()

    print()
    saved = sorted(OUT.glob("*.png"))
    print(f"Done — {len(saved)} screenshots saved to {OUT}/")
    for f in saved:
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")


if __name__ == "__main__":
    run()
