"""NFR-08 accessibility of the app page: every control is named, focusable and keyboard-operable."""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "glide" / "web"


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.found, self.imgs = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img":
            self.imgs.append(a)
        if tag in ("button", "input", "select", "textarea", "a"):
            self.stack.append([tag, a, ""])

    def handle_data(self, data):
        for item in self.stack:
            item[2] += data

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1][0] == tag:
            self.found.append(self.stack.pop())

    def close(self):
        super().close()
        self.found += self.stack                              # void elements (input) never close


class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")
        p = Controls()
        p.feed(cls.html)
        p.close()
        cls.controls, cls.imgs = p.found, p.imgs

    def test_every_control_has_a_name(self):
        for tag, a, text in self.controls:
            named = text.strip() or a.get("aria-label") or a.get("aria-labelledby") or a.get("title")
            self.assertTrue(named, f"unnamed <{tag}> {a}")

    def test_images_have_alt_text(self):
        for a in self.imgs:
            self.assertIn("alt", a)

    def test_no_control_is_taken_out_of_the_tab_order(self):
        self.assertNotRegex(self.html, r'tabindex="-1"')

    def test_switches_and_choices_use_aria_roles(self):
        self.assertGreaterEqual(len(re.findall(r'role="switch"', self.html)), 4)
        self.assertGreaterEqual(len(re.findall(r'role="radiogroup"', self.html)), 4)

    def test_every_full_window_flow_can_be_left_from_the_keyboard(self):
        js = (WEB / "app.js").read_text(encoding="utf-8")
        for flow in ("record", "setup", "pointing"):
            self.assertIn(f'id="{flow}"', self.html)
        self.assertIn('ev.key === "Escape"', js)
        self.assertIn("ptClose()", js)                         # pointing test: Esc / Enter
        self.assertIn('cmd("setup_cancel")', js)               # recalibration: Esc
        self.assertIn('cmd("record_close")', js)               # training result: Enter / Esc


if __name__ == "__main__":
    unittest.main()
