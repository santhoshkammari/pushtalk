#!/usr/bin/env python3
"""Voice HUD for pushtalk - Spotlight-style glass panel (PyQt5).

This is the Spotlight AI bar's look and window behaviour, re-pointed at the
voice bus instead of a text input:

  - a solid dark rounded panel (alpha 248 - barely translucent, clean)
  - a REAL managed top-level window: it shows in alt-tab, takes keyboard
    focus, and - unlike the old overlay - is NOT always-on-top, so you can
    bury it behind other apps and alt-tab back to it like any window
  - collapsed while you speak (just a slim waveform); expands when the
    agent's answer starts streaming; stays until you press Esc

There is no prompt box. The panel is driven entirely over the unix socket
(same wire format as ui_bus): one JSON line per event.

States (bus `state` field):
    level <0..1>   high-rate mic level - drives the waveform only, no layout
    listening      collapsed bar, live waveform
    thinking [txt] animated dots; if txt given (the transcript) it shows too
    delta <text>   the agent's answer, streamed in place, panel expanded
    tool <name>    small "running <tool>" hint under the answer
    reply <text>   answer settles, stays until Esc
    interrupted    red flash, answer dropped, back to collapsed
    error <msg>    red cross + message, stays until Esc
    idle           fade out

Run standalone (listens on the ui_bus socket), using the venv interpreter -
see run.sh (LIVE_ASR_VENV, defaults to ~/main):
    "$LIVE_ASR_VENV"/bin/python overlay_qt.py
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading

from markdown_it import MarkdownIt
from PyQt5.QtCore import (QEasingCurve, QPoint, QPropertyAnimation, QRect,
                          QRectF, Qt, QTimer, pyqtSignal)
from PyQt5.QtGui import (QColor, QFont, QIcon, QPainter, QPixmap, QPen,
                         QTextCursor)
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QHBoxLayout, QLabel,
                             QMenu, QSystemTrayIcon, QTextEdit, QVBoxLayout,
                             QWidget)

SOCKET_PATH = os.path.expanduser("~/.cache/pushtalk-ui.sock")
POS_PATH = os.path.expanduser("~/.cache/pushtalk-ui-pos.json")

WIDTH = 720
COLLAPSED_H = 64
EXPANDED_H = 420
ANIM_MS = 200
FADE_MS = 160

# Panel fill - Spotlight's value. 248/255 = ~97% opaque: a clean solid dark
# card with just a hint of the desktop bleeding through.
PANEL_RGBA = (28, 28, 30, 248)
BORDER_RGBA = (255, 255, 255, 30)


_MD = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])


def _md_html(text):
    """Markdown -> HTML for QTextEdit (Qt ignores table CSS, so set attributes)."""
    html = _MD.render(text)
    return html.replace("<table>", '<table border="1" cellspacing="0" cellpadding="6">')


class Viz(QWidget):
    """Slim waveform / thinking-dots / error-cross glyph.

    Sits at the left of the collapsed bar. Draws for listening / thinking /
    error; collapses to zero width for a plain reply so the answer text
    starts flush left.
    """

    DRAWS = ("listening", "thinking", "error")

    def __init__(self):
        super().__init__()
        self.setFixedSize(72, 22)
        self.state = "idle"
        self.phase = 0.0
        self.level = 0.0

    def set_state(self, state):
        self.state = state
        self.setFixedWidth(72 if state in self.DRAWS else 0)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2

        if self.state == "listening":
            bars, bw = 9, 3.0
            gap = (w - bars * bw) / (bars - 1)
            drive = 0.18 + 0.82 * self.level
            for i in range(bars):
                a = math.sin(self.phase * 3.4 + i * 0.62)
                amp = (0.16 + 0.84 * abs(a) * drive) * (h * 0.46)
                p.setBrush(QColor(125, 212, 255,
                                  int(255 * (0.45 + 0.55 * abs(a)))))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(QRectF(i * (bw + gap), mid - amp, bw, amp * 2),
                                  bw / 2, bw / 2)

        elif self.state == "thinking":
            for i in range(3):
                t = max(0.0, math.sin(self.phase * 3.0 - i * 0.9))
                r = 2.8 + 1.8 * t
                p.setBrush(QColor(158, 140, 255, int(255 * (0.35 + 0.65 * t))))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPoint(10 + i * 14, int(mid)), int(r), int(r))

        elif self.state == "error":
            p.setPen(QPen(QColor(255, 115, 115), 2.2, Qt.SolidLine,
                          Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(10, int(mid - 7), 26, int(mid + 7))
            p.drawLine(26, int(mid - 7), 10, int(mid + 7))
        p.end()


class Overlay(QWidget):
    """Spotlight-style bar, driven by the voice bus.

    Frameless, rounded, drag-to-move. A normal managed window (alt-tab,
    focus, NOT always-on-top).
    """

    event_in = pyqtSignal(str, str)   # marshals socket thread -> GUI thread

    def __init__(self):
        super().__init__()
        self.state = "idle"
        self._expanded = False
        self._drag_pos = None
        self.user_pos = self._load_pos()

        self._build_ui()
        self._place()

        self.anim = QTimer(self, interval=33, timeout=self._tick)
        self._fade = QPropertyAnimation(self, b"windowOpacity", duration=FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)

        self.event_in.connect(self.on_event, Qt.QueuedConnection)
        self.setWindowOpacity(0.0)

    # -- construction -----------------------------------------------------

    def _build_ui(self):
        # A normal top-level window: appears in alt-tab, takes focus.
        # Frameless + translucent for the rounded card, but deliberately NO
        # WindowStaysOnTopHint - other apps can cover it.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("Voice Assistant")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(0)

        # Top row: waveform / dots + the heard transcript (small, dim).
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.viz = Viz()
        row.addWidget(self.viz, 0, Qt.AlignVCenter)

        self.status = QLabel("")
        self.status.setFont(QFont("Inter", 8, QFont.Bold))
        self.status.setStyleSheet("color: rgba(127,212,255,0.85);"
                                  "letter-spacing: 2px; background: transparent;")
        row.addWidget(self.status, 0, Qt.AlignVCenter)

        self.heard = QLabel("")
        self.heard.setFont(QFont("SF Pro Text", 12))
        self.heard.setStyleSheet("color: #9aa0a6; background: transparent;")
        self.heard.setWordWrap(False)
        row.addWidget(self.heard, 1, Qt.AlignVCenter)
        lay.addLayout(row)

        # The agent's answer, streamed in place. Hidden until it expands.
        self.output = QTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setFrameShape(QTextEdit.NoFrame)
        self.output.setStyleSheet("""
            QTextEdit {
                background: transparent; border: none;
                border-top: 1px solid rgba(255,255,255,22);
                margin-top: 10px; padding: 12px 0 0 0;
                color: #E6E6E6;
                font-family: "SF Pro Text","Segoe UI","Ubuntu",sans-serif;
                font-size: 15px;
            }
            QScrollBar:vertical { width: 4px; background: transparent; }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,45); border-radius: 2px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        """)
        self.output.document().setDefaultStyleSheet(
            "code { font-family: monospace; background: #2a2a2a; }"
            "pre { background: #2a2a2a; }"
            "th { background: #2a2a2a; }"
            "a { color: #7fb0ff; }")
        self.output.setVisible(False)
        lay.addWidget(self.output)

        self.hint = QLabel("")
        self.hint.setFont(QFont("Inter", 8))
        self.hint.setStyleSheet("color: rgba(220,232,255,0.32);"
                                "background: transparent;")
        self.hint.setVisible(False)
        lay.addWidget(self.hint)

    def _place(self):
        if self.user_pos is not None:
            x, y = self._clamp(*self.user_pos)
            self.setGeometry(x, y, WIDTH, COLLAPSED_H)
            return
        d = QDesktopWidget()
        s = d.screenGeometry(d.screenNumber(d.cursor().pos()))
        self.setGeometry(
            s.x() + (s.width() - WIDTH) // 2,
            s.y() + s.height() // 4,
            WIDTH, COLLAPSED_H,
        )

    def _clamp(self, x, y):
        d = QDesktopWidget()
        s = d.screenGeometry(d.screenNumber(QPoint(x, y)))
        x = max(s.x(), min(x, s.x() + s.width() - WIDTH))
        y = max(s.y(), min(y, s.y() + s.height() - self.height()))
        return x, y

    # -- painting -------------------------------------------------------

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(*PANEL_RGBA))
        p.setPen(QColor(*BORDER_RGBA))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 14, 14)
        p.end()

    # -- bus events ---------------------------------------------------

    def on_event(self, state, text):
        if state == "level":
            try:
                self.viz.level = max(0.0, min(1.0, float(text)))
            except ValueError:
                pass
            return

        if state == "idle":
            self.dismiss()
            return

        was_idle = self.state == "idle"

        if state == "interrupted":
            self.state = "idle"
            self.viz.set_state("idle")
            self.status.setText("")
            self.heard.setText("")
            self.hint.setVisible(False)
            self.anim.stop()
            self._collapse()
            return

        if state in ("delta", "reply"):
            self.state = "reply"
            self.viz.set_state("reply")
            self.status.setText("")
            self.heard.setText("")
            self.anim.stop()
            self.output.setHtml(_md_html(text or "(no answer)"))
            self.output.moveCursor(QTextCursor.End)
            self._expand()
            self.hint.setVisible(False)
            if was_idle:
                self._show_focused()
            self._fade_to(1.0)
            return

        if state == "tool":
            self.hint.setText(f"  running  {text}")
            self.hint.setVisible(True)
            return

        if state == "error":
            self.state = "error"
            self.viz.set_state("error")
            self.status.setText("ERROR")
            self.heard.setText("")
            self.output.setPlainText(text or "(error)")
            self.anim.stop()
            self._expand()
            if was_idle:
                self._show_focused()
            self._fade_to(1.0)
            return

        if state in ("listening", "thinking"):
            self.state = state
            self.viz.set_state(state)
            self.status.setText("LISTENING" if state == "listening" else "THINKING")
            # `thinking` carries the transcript; `listening` never has text.
            self.heard.setText(_elide(text, 64) if state == "thinking" else "")
            self.hint.setVisible(False)
            self._collapse()
            self.anim.start()
            if was_idle:
                self._show_focused()
            self._fade_to(1.0)

    # -- expand / collapse ------------------------------------------

    def _expand(self):
        if self._expanded:
            return
        self._expanded = True
        self.output.setVisible(True)
        self._animate_h(self.height(), EXPANDED_H)

    def _collapse(self):
        if not self._expanded:
            # still make sure height is the collapsed one
            if self.height() != COLLAPSED_H:
                self.setFixedHeight(COLLAPSED_H)
                self.setMinimumHeight(0)
                self.setMaximumHeight(16777215)
                self.resize(self.width(), COLLAPSED_H)
            return
        self._expanded = False
        self._animate_h(self.height(), COLLAPSED_H, after=self._after_collapse)

    def _after_collapse(self):
        self.output.setVisible(False)
        self.output.clear()
        self.hint.setVisible(False)

    def _animate_h(self, h0, h1, after=None):
        x, y, w = self.x(), self.y(), self.width()
        self._a = QPropertyAnimation(self, b"geometry")
        self._a.setDuration(ANIM_MS)
        self._a.setStartValue(QRect(x, y, w, h0))
        self._a.setEndValue(QRect(x, y, w, h1))
        self._a.setEasingCurve(QEasingCurve.OutCubic)
        if after:
            self._a.finished.connect(after)
        self._a.start()

    # -- show / hide / fade ---------------------------------------

    def _show_focused(self):
        self._place()
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)

    def _fade_to(self, target, then=None):
        self._fade.stop()
        try:
            self._fade.finished.disconnect()
        except TypeError:
            pass
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(target)
        if then:
            self._fade.finished.connect(then)
        self._fade.start()

    def dismiss(self):
        self.state = "idle"
        self.viz.set_state("idle")
        self.anim.stop()
        self._expanded = False
        self.output.setVisible(False)
        self.output.clear()
        self.hint.setVisible(False)
        self.status.setText("")
        self.heard.setText("")

        def done():
            self.hide()
            self.setFixedHeight(COLLAPSED_H)
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.resize(self.width(), COLLAPSED_H)
        self._fade_to(0.0, done)

    # -- animation tick ------------------------------------------

    def _tick(self):
        self.viz.phase += 0.085
        if self.viz.state == "listening":
            self.viz.level *= 0.88
        self.viz.update()

    # -- keyboard / drag ----------------------------------------

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.dismiss()
            e.accept()
            return
        super().keyPressEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag_pos and e.buttons() & Qt.LeftButton:
            self.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, _):
        if self._drag_pos is not None:
            x, y = self._clamp(self.x(), self.y())
            self.move(x, y)
            self.user_pos = (x, y)
            self._save_pos()
        self._drag_pos = None
        self.setCursor(Qt.OpenHandCursor)

    def closeEvent(self, e):
        # Never destroy; just hide - the Bus keeps feeding events.
        e.ignore()
        self.dismiss()

    # -- position persistence -------------------------------------

    def _load_pos(self):
        try:
            with open(POS_PATH) as f:
                d = json.load(f)
            return (int(d["x"]), int(d["y"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_pos(self):
        if self.user_pos is None:
            return
        try:
            os.makedirs(os.path.dirname(POS_PATH), exist_ok=True)
            with open(POS_PATH, "w") as f:
                json.dump({"x": self.user_pos[0], "y": self.user_pos[1]}, f)
        except OSError:
            pass

    def reset_position(self):
        self.user_pos = None
        try:
            os.unlink(POS_PATH)
        except OSError:
            pass
        self._place()


def _elide(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


class Bus(threading.Thread):
    """Datagram socket -> Qt signal. Same wire format as ui_bus."""

    daemon = True

    def __init__(self, emit):
        super().__init__()
        self.emit = emit

    def run(self):
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        srv.bind(SOCKET_PATH)
        print(f"overlay listening on {SOCKET_PATH}", file=sys.stderr, flush=True)
        while True:
            try:
                data, _ = srv.recvfrom(65536)
                msg = json.loads(data.decode())
                self.emit(msg.get("state", ""), msg.get("text", ""))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue


def make_tray(app, overlay):
    icon = QIcon.fromTheme("audio-input-microphone")
    if icon.isNull():
        pm = QPixmap(22, 22)
        pm.fill(QColor(127, 212, 255))
        icon = QIcon(pm)
    tray = QSystemTrayIcon(icon, app)
    tray.setToolTip("Voice Assistant")

    menu = QMenu()
    last = {"reply": ""}
    menu.addAction("Show last reply").triggered.connect(
        lambda: last["reply"] and overlay.on_event("reply", last["reply"]))
    menu.addAction("Reset position").triggered.connect(overlay.reset_position)
    menu.addSeparator()
    menu.addAction("Quit").triggered.connect(app.quit)
    tray.setContextMenu(menu)
    tray.show()
    return tray, last


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay()
    tray, last = make_tray(app, overlay)

    def emit(state, text):
        if state == "reply" and text:
            last["reply"] = text
        overlay.event_in.emit(state, text)

    Bus(emit).start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
