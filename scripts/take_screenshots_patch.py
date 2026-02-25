"""Patch script — captures the missing screenshots."""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, Page

BASE = "http://localhost:5173"
OUT  = Path("screenshots")
VIEWPORT = {"width": 1440, "height": 900}


def wait_idle(page: Page, timeout=8000):
    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception:
        pass


def ss(page: Page, filename: str, extra_wait=0.5):
    page.evaluate("window.scrollTo(0,0)")
    time.sleep(extra_wait)
    page.screenshot(path=str(OUT / filename), full_page=False)
    print(f"  ✓ {filename}")


def run():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(viewport=VIEWPORT)
        page = ctx.new_page()

        # Login
        page.goto(f"{BASE}/login", wait_until="domcontentloaded")
        wait_idle(page)
        time.sleep(1)
        page.fill("input[type='text']", "admin")
        page.fill("input[type='password']", "password")
        page.click("button[type='submit']")
        wait_idle(page, 8000)
        time.sleep(2)
        print("Logged in")

        # ── GraphSAGE How It Works (drawer closed) ────────────────────────────
        print("GraphSAGE How It Works tab...")
        page.goto(f"{BASE}/graphsage", wait_until="domcontentloaded")
        wait_idle(page, 12000)
        time.sleep(4)
        # Click the How It Works tab (drawer is NOT open)
        try:
            howto = page.locator("button:has-text('How It Works')").first
            howto.click(timeout=5000)
            time.sleep(2)
            wait_idle(page, 4000)
            # Scroll down to show the mule patterns
            page.evaluate("window.scrollTo(0, 300)")
            time.sleep(1)
            page.screenshot(path=str(OUT / "graphsage-howto.png"), full_page=False)
            print("  ✓ graphsage-howto.png")
        except Exception as e:
            print(f"  ✗ howto failed: {e}")

        # ── Transaction detail ────────────────────────────────────────────────
        print("Transaction detail...")
        page.goto(f"{BASE}/transactions", wait_until="domcontentloaded")
        wait_idle(page, 12000)
        time.sleep(3)
        # Click first row
        try:
            rows = page.locator("tbody tr")
            count = rows.count()
            print(f"  Found {count} rows")
            if count > 0:
                rows.first.click(timeout=5000)
                wait_idle(page, 6000)
                time.sleep(2.5)
                ss(page, "transaction-detail.png", extra_wait=1)
        except Exception as e:
            print(f"  ✗ detail failed: {e}")

        # ── Customer detail (click first customer) ────────────────────────────
        print("Customer detail...")
        page.goto(f"{BASE}/customers", wait_until="domcontentloaded")
        wait_idle(page, 15000)
        time.sleep(5)
        try:
            # Look for any clickable row / link in customer list
            rows = page.locator("tbody tr")
            cnt = rows.count()
            print(f"  Found {cnt} customer rows")
            if cnt > 0:
                rows.first.click(timeout=5000)
                wait_idle(page, 8000)
                time.sleep(3)
                ss(page, "customer-detail.png", extra_wait=1)
        except Exception as e:
            print(f"  ✗ customer detail failed: {e}")

        # ── Feature store expanded ────────────────────────────────────────────
        print("Feature store expanded row...")
        page.goto(f"{BASE}/feature-store", wait_until="domcontentloaded")
        wait_idle(page, 10000)
        time.sleep(3)
        try:
            # Try different expand button selectors
            for sel in [
                "button:has-text('▶')",
                "button:has-text('Details')",
                "button[aria-label*='expand']",
                "tbody tr td:first-child button",
                "tbody tr:first-child button",
            ]:
                try:
                    btn = page.locator(sel).first
                    if btn.count() > 0:
                        btn.click(timeout=2000)
                        time.sleep(1.5)
                        ss(page, "feature-store-expanded.png")
                        break
                except Exception:
                    pass
        except Exception as e:
            print(f"  ✗ feature expand failed: {e}")

        # ── Submit transaction + result ───────────────────────────────────────
        print("Submit transaction result...")
        page.goto(f"{BASE}/transactions/new", wait_until="domcontentloaded")
        wait_idle(page, 8000)
        time.sleep(2)
        ss(page, "submit-transaction.png")

        browser.close()

    print()
    saved = sorted(OUT.glob("*.png"))
    print(f"Total: {len(saved)} screenshots in {OUT}/")
    for f in saved:
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")


if __name__ == "__main__":
    run()
