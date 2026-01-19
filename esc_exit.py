# esc_exit.py
from __future__ import annotations

def esc_pressed() -> bool:
    """
    Windows console key check.
    Works when the console window has focus.
    No external dependencies.
    """
    try:
        import msvcrt  # Windows only
    except Exception:
        return False

    # Drain key buffer; return True if ESC detected
    pressed = False
    while msvcrt.kbhit():
        ch = msvcrt.getwch()  # wide char
        if ch == "\x1b":      # ESC
            pressed = True
    return pressed
