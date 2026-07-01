#!/usr/bin/env python3
# annotate_windows_tk.py
import json, sys, math
from pathlib import Path
from typing import List, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox

import os
DEFAULT_JSON_NAME = "_windows.json"

try:
    from PIL import Image, ImageTk
except Exception:
    print("Please: pip install pillow")
    sys.exit(1)

HELP_TEXT = """keys:
  navigation
    → / d : next frame (+step)         ← / a : prev frame (-step)
    ]     : increase step               [     : decrease step
    D     : +10×step                    A     : -10×step
    Home  : first frame                 End   : last frame
    g     : go to timestamp (ns)        t     : go to frame index (1-based)

  window labeling
    digits 0-9 : build current label    backspace : clear label
    s          : mark start at current  e        : mark end at current
    SPACE      : commit window (needs label + start + end)
    u          : undo last committed    r        : clear marks

  file
    w     : write JSON now (to <framedir>/<json_name>)
    f     : set JSON filename (same folder)
    q/ESC : save & quit
    h     : toggle help overlay
"""


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def load_frames_sorted(framedir: Path) -> Tuple[List[Path], List[int]]:
    files = [p for p in framedir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not files:
        raise SystemExit(f"No image files found in {framedir}")
    def parse_ns(p: Path):
        try: return int(p.stem)
        except: return math.inf
    files = sorted(files, key=parse_ns)
    files = [p for p in files if p.stem.isdigit()]
    if not files:
        raise SystemExit("No timestamp-named images (integer stems) found.")
    ts = [int(p.stem) for p in files]
    return files, ts

class App(tk.Tk):
    def __init__(self, framedir: Path):
        super().__init__()
        self.json_name = DEFAULT_JSON_NAME
        self.title("Quick Window Annotator (Tk)")
        self.geometry("1280x860")
        self.configure(bg="#111")

        self.framedir = framedir
        self.files, self.ts = load_frames_sorted(self.framedir)
        self.N = len(self.files)

        # state
        self.i = 0
        self.step = 1
        self.show_help = False
        self.label_digits = ""
        self.start_ns = None
        self.end_ns = None
        self.windows = {}   # label(int)-> {"start_ns":..., "end_ns":...}
        self.history = []   # for undo

        # UI
        self.status = tk.StringVar(value=f"{self.N} frames loaded")
        self._build_ui()
        self._bind_keys()
        self._render()

        

    # ---------- UI ----------
    def _build_ui(self):
        top = tk.Frame(self, bg="#111"); top.pack(fill="x")
        tk.Label(top, textvariable=self.status, fg="#ddd", bg="#111", anchor="w").pack(side="left", padx=10, pady=6)

        self.canvas = tk.Canvas(self, bg="#1f1f1f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._render())

    def _bind_keys(self):
        # quit/save/help
        self.bind("f", lambda e: self._prompt_set_json_name())
        self.bind("q", lambda e: self._save_and_quit())
        self.bind("<Escape>", lambda e: self._save_and_quit())
        self.bind("w", lambda e: self._save_json())
        self.bind("h", lambda e: self._toggle_help())

        # nav
        self.bind("d", lambda e: self._jump(+self.step))
        self.bind("a", lambda e: self._jump(-self.step))
        self.bind("<Right>", lambda e: self._jump(+self.step))
        self.bind("<Left>",  lambda e: self._jump(-self.step))
        self.bind("D", lambda e: self._jump(+10*self.step))
        self.bind("A", lambda e: self._jump(-10*self.step))
        self.bind("<Home>", lambda e: self._goto_index(0))
        self.bind("<End>",  lambda e: self._goto_index(self.N-1))
        self.bind("]", lambda e: self._set_step(self.step*2))
        self.bind("[", lambda e: self._set_step(max(1, self.step//2)))
        self.bind("t", lambda e: self._prompt_goto_index())
        self.bind("g", lambda e: self._prompt_goto_ts())

        # labels & windows
        for d in "0123456789":
            self.bind(d, self._on_digit)
        self.bind("<BackSpace>", lambda e: self._clear_label())
        self.bind("s", lambda e: self._mark_start())
        self.bind("e", lambda e: self._mark_end())
        self.bind("r", lambda e: self._clear_marks())
        self.bind("<space>", lambda e: self._commit_window())
        self.bind("u", lambda e: self._undo())

    # ---------- Actions ----------
    def _jump(self, di: int):
        self.i = max(0, min(self.N-1, self.i + di))
        self._render()

    def _goto_index(self, idx: int):
        self.i = max(0, min(self.N-1, idx))
        self._render()

    def _set_step(self, s: int):
        self.step = max(1, min(self.N, int(s)))
        self._render_status()

    def _prompt_goto_index(self):
        val = self._ask("Go to frame index (1..{}):".format(self.N), default=str(self.i+1))
        if val is None: return
        try:
            idx = int(val) - 1
            self._goto_index(idx)
        except:
            messagebox.showwarning("Index", "Please input an integer in range.")

    def _prompt_set_json_name(self):
        name = self._ask("Set JSON filename (same folder):", default=self.json_name)
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showwarning("Filename", "Filename cannot be empty.")
            return
        if os.sep in name or (os.altsep and os.altsep in name):
            messagebox.showwarning("Filename", "Do not include directory separators.")
            return
        if not name.lower().endswith(".json"):
            name += ".json"
        self.json_name = name
        self._render_status()
    
    def _prompt_goto_ts(self):
        val = self._ask("Go to timestamp (ns):", default=str(self.ts[self.i]))
        if val is None: return
        try:
            tgt = int(val)
            # binary search
            import bisect
            j = bisect.bisect_left(self.ts, tgt)
            if j >= self.N: j = self.N-1
            self._goto_index(j)
        except:
            messagebox.showwarning("Timestamp", "Please input an integer timestamp (ns).")

    def _on_digit(self, e):
        if len(self.label_digits) < 9:
            self.label_digits += e.char
            self._render()

    def _clear_label(self):
        self.label_digits = ""
        self._render()

    def _mark_start(self):
        self.start_ns = int(self.ts[self.i])
        self._render()

    def _mark_end(self):
        self.end_ns = int(self.ts[self.i])
        self._render()

    def _clear_marks(self):
        self.start_ns = None; self.end_ns = None
        self._render()

    def _commit_window(self):
        if self.label_digits == "":
            messagebox.showinfo("Label", "Type digits to set an integer label first.")
            return
        try:
            lbl = int(self.label_digits)
        except:
            messagebox.showinfo("Label", "Label must be an integer.")
            return
        if self.start_ns is None or self.end_ns is None:
            messagebox.showinfo("Window", "Mark both start (s) and end (e).")
            return
        s, e = int(self.start_ns), int(self.end_ns)
        if e < s: s, e = e, s
        self.windows[lbl] = {"start_ns": s, "end_ns": e}
        self.history.append((lbl, s, e))
        # keep label for quick repeats, clear marks
        self.start_ns = None; self.end_ns = None
        self._render()

    def _undo(self):
        if not self.history: return
        lbl, s, e = self.history.pop()
        if self.windows.get(lbl, {}) == {"start_ns": s, "end_ns": e}:
            del self.windows[lbl]
        self._render()

    def _toggle_help(self):
        self.show_help = not self.show_help
        self._render()

    # ---------- Render ----------
    def _render_status(self):
        info = (
            f"frame {self.i+1}/{self.N}  step={self.step}  ts={self.ts[self.i]}  "
            f"label={self.label_digits or '(none)'}"
        )
        s_txt = (
            f"{info}   start_ns={self.start_ns if self.start_ns is not None else '(unset)'}   "
            f"end_ns={self.end_ns if self.end_ns is not None else '(unset)'}   "
            f"windows={len(self.windows)}   json={self.json_name}"
        )
        self.status.set(s_txt)

    def _render(self):
        self.canvas.delete("all")
        if self.N == 0: return
        self._render_status()

        # load current image
        path = self.files[self.i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showwarning("Image", f"Failed to read {path.name}: {e}")
            return

        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        iw, ih = img.size
        scale = min(cw/iw, ch/ih)
        new_size = (max(1,int(iw*scale)), max(1,int(ih*scale)))
        img = img.resize(new_size, Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        x = (cw - new_size[0])//2
        y = (ch - new_size[1])//2
        self.canvas.create_image(x, y, anchor="nw", image=self._photo)

        # overlays (text)
        overlay = [
            f"idx {self.i+1}/{self.N}  step={self.step}",
            f"ts [ns]: {self.ts[self.i]}",
            f"label: {self.label_digits or '(none)'}",
            f"start_ns: {self.start_ns if self.start_ns is not None else '(unset)'}",
            f"end_ns:   {self.end_ns if self.end_ns is not None else '(unset)'}",
            f"json file: {self.json_name}",
            f"windows: {len(self.windows)}   (h for help)"
        ]
        self._draw_text_block(10, 20, overlay)

        if self.show_help:
            self._draw_text_block(10, 180, HELP_TEXT.splitlines())

        # show committed windows list (right side)
        if self.windows:
            rows = [f"{k}: {v['start_ns']} .. {v['end_ns']}" for k,v in sorted(self.windows.items())]
            self._draw_text_block(cw-420, 20, ["Committed windows:"] + rows, anchor="nw")

    def _draw_text_block(self, x, y, lines, anchor="nw"):
        # simple white text with shadow
        for k, line in enumerate(lines):
            yy = y + k*20
            self.canvas.create_text(x+1, yy+1, text=line, fill="#000", anchor=anchor, font=("TkDefaultFont", 11, "bold"))
            self.canvas.create_text(x,   yy,   text=line, fill="#fff", anchor=anchor, font=("TkDefaultFont", 11, "bold"))

    # ---------- Save / Quit ----------
    def _save_json(self):
        out = {str(k): {"start_ns": int(v["start_ns"]), "end_ns": int(v["end_ns"])} for k, v in self.windows.items()}
        out_path = self.framedir / self.json_name
        try:
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            self.status.set(f"[saved] {out_path}")
        except Exception as e:
            messagebox.showerror("Save JSON", str(e))

    def _save_and_quit(self):
        self._save_json()
        self.destroy()

    # ---------- Small prompt ----------
    def _ask(self, title: str, default: str = "") -> str | None:
        dlg = tk.Toplevel(self)
        dlg.title("Input")
        dlg.transient(self)
        dlg.grab_set()
        tk.Label(dlg, text=title).pack(padx=10, pady=(10,4))
        var = tk.StringVar(value=default)
        e = tk.Entry(dlg, textvariable=var, width=36)
        e.pack(padx=10, pady=4)
        e.focus_set()
        ans = {"val": None}
        def ok():
            ans["val"] = var.get()
            dlg.destroy()
        def cancel():
            dlg.destroy()
        btns = tk.Frame(dlg); btns.pack(pady=8)
        tk.Button(btns, text="OK", width=8, command=ok).pack(side="left", padx=6)
        tk.Button(btns, text="Cancel", width=8, command=cancel).pack(side="left", padx=6)
        self.wait_window(dlg)
        return ans["val"]

def choose_directory() -> Path:
    root = tk.Tk(); root.withdraw()
    p = filedialog.askdirectory(title="Select frame directory")
    root.destroy()
    if not p: sys.exit("No directory selected.")
    return Path(p)

if __name__ == "__main__":
    # simple argv parsing: first non-flag is framedir; optional: --json <name>
    framedir = None
    json_name_cli = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--json" and i + 1 < len(args):
            json_name_cli = args[i + 1]
            i += 2
        elif a.startswith("--"):
            print(f"Unknown option: {a}")
            sys.exit(2)
        else:
            # first positional = framedir
            if framedir is None:
                framedir = Path(a)
            else:
                print(f"Ignoring extra positional argument: {a}")
            i += 1

    if framedir is None:
        framedir = choose_directory()
    if not framedir.exists():
        print(f"Not found: {framedir}"); sys.exit(1)

    app = App(framedir)
    if json_name_cli:
        # sanitize like the prompt handler
        name = json_name_cli.strip()
        if name:
            if os.sep in name or (os.altsep and os.altsep in name):
                print("Error: --json name must not include directory separators.")
                sys.exit(2)
            if not name.lower().endswith(".json"):
                name += ".json"
            app.json_name = name
    app.mainloop()