#!/usr/bin/env python3
"""
Interaction Window Verifier (lean)

What it does:
- Loads an interaction-splitting JSON file (keys like window_0, window_1, ...)
- Lets you pick a frame directory whose files are named by **nanosecond** timestamps
- For each window, shows N=5 evenly spaced thumbnails (nearest-timestamp match, robust to tiny timing offsets)
- Lets you:
  • Toggle "Confirmed" (valid window)
  • Select "Open" or "Close"
  • Enter an integer "Index"
- Saves only confirmed windows to "<original_stem>_confirmed.json" next to the source JSON
  (auto-increments to _v2, _v3, ... if the file already exists)

Hotkeys:
- ← / → : previous / next window
- Space / Enter : toggle Confirm (unless you’re typing in a text field)
- o / c : set Open / Close
- i     : focus the Index field
"""

import json
import re
import bisect
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NUMBER_RE  = re.compile(r'([-+]?\d+(?:\.\d+)?)')


@dataclass
class Window:
    name: str
    start_ns: int
    end_ns: int
    duration_s: float


class FrameIndex:
    """
    Index frames by numeric timestamp parsed from filename (first number found).
    Assumes filenames are **nanoseconds**. We sort numerically, not lexicographically.
    """
    def __init__(self, frame_dir: Path):
        self.frame_dir = Path(frame_dir)
        self.ts_list: List[float] = []  # numeric timestamps (ns, as float for bisect)
        self.files: List[Path] = []
        self._index()

    def _index(self):
        if not self.frame_dir.exists():
            raise FileNotFoundError(f"Frame dir does not exist: {self.frame_dir}")
        pairs = []  # (ts_numeric, path)
        for pth in self.frame_dir.iterdir():
            if pth.suffix.lower() in IMAGE_EXTS and pth.is_file():
                m = NUMBER_RE.search(pth.stem)
                if not m:
                    continue
                ts_str = m.group(1)
                # timestamps are nanoseconds (integers)
                try:
                    ts_val = int(ts_str)
                except Exception:
                    # fallback to float if needed
                    try:
                        ts_val = float(ts_str)
                    except Exception:
                        continue
                pairs.append((float(ts_val), pth))
        if not pairs:
            exts = ", ".join(sorted(IMAGE_EXTS))
            raise RuntimeError(f"No timestamped image files found in {self.frame_dir} ({exts})")
        pairs.sort(key=lambda x: x[0])
        self.ts_list = [ts for ts, _ in pairs]
        self.files   = [fp for _, fp in pairs]

    def nearest(self, target_ns: float) -> Tuple[float, Path]:
        """
        target_ns: timestamp in nanoseconds (float)
        returns (timestamp_ns, path) of nearest frame
        """
        idx = bisect.bisect_left(self.ts_list, target_ns)
        if idx == 0:
            return self.ts_list[0], self.files[0]
        if idx >= len(self.ts_list):
            return self.ts_list[-1], self.files[-1]
        before = self.ts_list[idx - 1]
        after  = self.ts_list[idx]
        if abs(after - target_ns) < abs(target_ns - before):
            return after, self.files[idx]
        else:
            return before, self.files[idx - 1]


def load_windows(json_path: Path) -> List[Window]:
    with open(json_path, "r") as f:
        data = json.load(f)
    windows: List[Window] = []
    # Preserve order by sorting keys like "window_0", "window_1" by trailing number
    def key_to_num(s: str) -> int:
        nums = re.findall(r'\d+', s)
        return int(nums[-1]) if nums else 0
    for k in sorted(data.keys(), key=key_to_num):
        w = data[k]
        windows.append(Window(
            name=k,
            start_ns=int(w["start_ns"]),
            end_ns=int(w["end_ns"]),
            duration_s=float(w["duration_s"]),
        ))
    return windows


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interaction Window Verifier (lean)")
        self.geometry("1000x760")

        self.json_path: Optional[Path] = None
        self.frame_dir: Optional[Path] = None
        self.windows: List[Window] = []
        self.idx: int = 0
        self.N: int = 5
        self.frame_index: Optional[FrameIndex] = None

        # timestamps are always ns
        self.unit_scale: float = 1.0

        # Per-window state: name -> payload
        self.confirmed_map: Dict[str, Dict] = {}

        # UI State
        self.single_var = tk.StringVar(value="")       # "open" / "close" / ""
        self.idx_var    = tk.StringVar()               # integer
        self.confirm_var = tk.BooleanVar(value=False)  # toggle confirmed
        self.idx_entry  = None                         # will store Entry widget ref

        self._build_ui()
        self._bind_keys()

        if not PIL_OK:
            messagebox.showwarning(
                "Pillow missing",
                "Pillow (PIL) is not installed. Images may not display.\n\n"
                "Install with: pip install pillow"
            )

    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self.json_var = tk.StringVar()
        self.dir_var  = tk.StringVar()
        self.n_var    = tk.StringVar(value=str(self.N))

        ttk.Label(top, text="Split JSON:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.json_var, width=50).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse", command=self.browse_json).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="Frame Dir:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.dir_var, width=50).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse", command=self.browse_dir).grid(row=1, column=2, padx=4)

        ttk.Label(top, text="Samples N:").grid(row=0, column=3, sticky="e", padx=(20, 4))
        ttk.Entry(top, textvariable=self.n_var, width=5).grid(row=0, column=4, sticky="w")
        ttk.Button(top, text="Load", command=self.load_all).grid(row=1, column=3, columnspan=2, sticky="we")

        top.columnconfigure(1, weight=1)

        # Info bar
        info = ttk.Frame(self, padding=8)
        info.pack(fill="x")
        self.info_label  = ttk.Label(info, text="No data loaded.")
        self.count_label = ttk.Label(info, text="")
        self.info_label.pack(side="left")
        self.count_label.pack(side="right")

        # Canvas for thumbnails
        self.canvas = tk.Canvas(self, bg="#202225", height=480)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas_images = []  # keep references so Tk doesn't GC them

        # Bottom controls (lean)
        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x", padx=8, pady=(0, 8))

        # Open/Close radio
        ttk.Label(bottom, text="Type:").grid(row=0, column=0, sticky="e")
        rb_open  = ttk.Radiobutton(bottom, text="Open",  value="open",  variable=self.single_var)
        rb_close = ttk.Radiobutton(bottom, text="Close", value="close", variable=self.single_var)
        rb_open.grid(row=0, column=1, sticky="w", padx=(6, 6))
        rb_close.grid(row=0, column=2, sticky="w", padx=(0, 12))

        # Index (int)
        ttk.Label(bottom, text="Index (int):").grid(row=0, column=3, sticky="e")
        self.idx_entry = ttk.Entry(bottom, textvariable=self.idx_var, width=10)
        self.idx_entry.grid(row=0, column=4, sticky="w", padx=(6, 0))

        # Confirm toggle + nav + save
        self.confirm_chk = ttk.Checkbutton(
            bottom, text="Confirmed", variable=self.confirm_var, command=self.toggle_confirm
        )
        self.confirm_chk.grid(row=1, column=0, sticky="w", pady=6)

        ttk.Button(bottom, text="⏮ Prev", command=self.prev_window).grid(row=1, column=1, sticky="we", pady=6)
        ttk.Button(bottom, text="⏭ Next", command=self.next_window).grid(row=1, column=2, sticky="we", pady=6)
        ttk.Button(bottom, text="💾 Save JSON", command=self.save_json).grid(row=1, column=3, sticky="we", pady=6)

        bottom.columnconfigure(2, weight=1)

        # Status
        self.status = ttk.Label(self, text="")
        self.status.pack(fill="x", padx=8, pady=(0, 8))

    def _bind_keys(self):
        self.bind("<Left>",  lambda e: self.prev_window())
        self.bind("<Right>", lambda e: self.next_window())

        # Return & Space toggle confirm unless typing in an Entry
        self.bind("<Return>", self._on_enter)
        self.bind("<space>",  self._on_space)

        # Hotkeys for type + focus index
        self.bind("o", lambda e: self.single_var.set("open"))
        self.bind("c", lambda e: self.single_var.set("close"))
        self.bind("i", lambda e: (self.idx_entry.focus_set() if self.idx_entry else None))

    def _on_space(self, event):
        w = self.focus_get()
        try:
            klass = w.winfo_class()
        except Exception:
            klass = ""
        if klass in ("TEntry", "Entry", "Text"):
            return  # let space insert in text fields
        self.confirm_var.set(not self.confirm_var.get())
        self.toggle_confirm()

    def _on_enter(self, event):
        w = self.focus_get()
        try:
            klass = w.winfo_class()
        except Exception:
            klass = ""
        if klass in ("TEntry", "Entry", "Text"):
            return  # let enter act in text fields
        self.confirm_var.set(not self.confirm_var.get())
        self.toggle_confirm()

    # --- File pickers ---
    def browse_json(self):
        p = filedialog.askopenfilename(
            title="Select interaction splitting JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if p:
            self.json_var.set(p)

    def browse_dir(self):
        d = filedialog.askdirectory(title="Select frame directory")
        if d:
            self.dir_var.set(d)

    # --- Loading ---
    def load_all(self):
        try:
            self.N = max(1, int(self.n_var.get()))
        except Exception:
            messagebox.showerror("Invalid N", "Samples N must be an integer ≥ 1.")
            return

        jp = Path(self.json_var.get())
        dp = Path(self.dir_var.get())
        if not jp.exists():
            messagebox.showerror("Missing JSON", f"JSON not found:\n{jp}")
            return
        if not dp.exists():
            messagebox.showerror("Missing frame dir", f"Directory not found:\n{dp}")
            return

        try:
            self.windows = load_windows(jp)
        except Exception as e:
            messagebox.showerror("JSON error", f"Failed to read windows:\n{e}")
            return

        try:
            self.frame_index = FrameIndex(dp)
        except Exception as e:
            messagebox.showerror("Frame index error", str(e))
            return

        self.json_path = jp
        self.frame_dir = dp
        self.idx = 0
        self.confirmed_map.clear()
        self.single_var.set("")
        self.idx_var.set("")
        self.confirm_var.set(False)

        self.update_info()
        self.show_current()

    def update_info(self):
        if not self.windows:
            self.info_label.config(text="No data loaded.")
            self.count_label.config(text="")
            return
        w = self.windows[self.idx]
        self.info_label.config(
            text=f"{w.name}: start={w.start_ns} ns, end={w.end_ns} ns, dur={w.duration_s:.3f}s  (timestamps assumed ns)"
        )
        self.count_label.config(text=f"Window {self.idx+1}/{len(self.windows)} | Confirmed: {len(self.confirmed_map)}")

    # --- Navigation ---
    def prev_window(self):
        if not self.windows:
            return
        self._cache_current_if_confirmed()
        self.idx = max(0, self.idx - 1)
        self.update_info()
        self.show_current()

    def next_window(self):
        if not self.windows:
            return
        self._cache_current_if_confirmed()
        self.idx = min(len(self.windows) - 1, self.idx + 1)
        self.update_info()
        self.show_current()

    def _cache_current_if_confirmed(self):
        if not self.windows:
            return
        w = self.windows[self.idx]
        if self.confirm_var.get():
            payload = self._current_payload(w)
            self.confirmed_map[w.name] = payload

    # --- Display ---
    def show_current(self):
        self.canvas.delete("all")
        self.canvas_images.clear()
        if not self.windows or not self.frame_index:
            return

        w = self.windows[self.idx]

        # Restore per-window state (if confirmed)
        if w.name in self.confirmed_map:
            saved = self.confirmed_map[w.name]
            self.single_var.set(saved.get("single_word_description", ""))
            self.idx_var.set("" if saved.get("index") is None else str(saved.get("index")))
            self.confirm_var.set(True)
        else:
            self.single_var.set("")
            self.idx_var.set("")
            self.confirm_var.set(False)

        # Generate targets and pick nearest frames
        start = float(w.start_ns) * self.unit_scale
        end   = float(w.end_ns)   * self.unit_scale
        targets = linspace(start, end, self.N)
        picks = []
        used_paths = set()
        for t in targets:
            ts, path = self.frame_index.nearest(t)
            if path in used_paths:
                i = bisect.bisect_left(self.frame_index.ts_list, ts)
                left  = i - 1 if i - 1 >= 0 else None
                right = i + 1 if i + 1 < len(self.frame_index.ts_list) else None
                if left is not None and self.frame_index.files[left] not in used_paths:
                    ts, path = self.frame_index.ts_list[left], self.frame_index.files[left]
                elif right is not None and self.frame_index.files[right] not in used_paths:
                    ts, path = self.frame_index.ts_list[right], self.frame_index.files[right]
            used_paths.add(path)
            picks.append((t, ts, path))

        # Layout horizontally
        W = self.canvas.winfo_width()  or 900
        H = self.canvas.winfo_height() or 480
        pad = 10
        cols = len(picks)
        thumb_w = max(120, min(280, (W - pad*(cols+1)) // cols))
        thumb_h = int(thumb_w * 9/16)

        x = pad
        y = pad

        for t_target, ts_found, p in picks:
            self.canvas.create_rectangle(x-4, y-4, x+thumb_w+4, y+thumb_h+58, outline="#444", width=1)

            if PIL_OK:
                try:
                    im = Image.open(p)
                    im.thumbnail((thumb_w, thumb_h))
                    photo = ImageTk.PhotoImage(im)
                    self.canvas.create_image(x, y, anchor="nw", image=photo)
                    self.canvas_images.append(photo)
                except Exception as e:
                    self.canvas.create_text(x+8, y+8, anchor="nw", fill="#ddd",
                                            text=f"Failed to load:\n{p.name}\n{e}")
            else:
                self.canvas.create_text(x+8, y+8, anchor="nw", fill="#ddd",
                                        text=f"Pillow not installed.\nCannot show {p.name}")

            delta = abs(ts_found - t_target)
            cap = f"{p.name}\nnearest Δ={delta:.0f} ns"
            self.canvas.create_text(x, y+thumb_h+8, anchor="nw", fill="#ccc",
                                    text=cap, font=("TkDefaultFont", 10))

            x += thumb_w + pad

    # --- Confirm toggle & Save ---
    def _current_payload(self, w: Window) -> Dict:
        single_choice = self.single_var.get().strip().lower()
        idx_txt = (self.idx_var.get() or "").strip()
        try:
            index_val = int(idx_txt) if idx_txt else None
        except Exception:
            messagebox.showerror("Invalid index", "Index must be an integer.")
            if w.name in self.confirmed_map:
                index_val = self.confirmed_map[w.name].get("index", None)
            else:
                index_val = None

        return {
            "name":       w.name,
            "start_ns":   w.start_ns,
            "end_ns":     w.end_ns,
            "duration_s": w.duration_s,
            "single_word_description": single_choice,  # 'open'/'close' or ''
            "index": index_val,
        }

    def toggle_confirm(self):
        if not self.windows:
            return
        w = self.windows[self.idx]
        if self.confirm_var.get():
            payload = self._current_payload(w)
            self.confirmed_map[w.name] = payload
            self.status.config(text=f"✔ Confirmed {w.name} ({len(self.confirmed_map)} total).")
        else:
            if w.name in self.confirmed_map:
                del self.confirmed_map[w.name]
            self.status.config(text=f"✖ Unconfirmed {w.name}. ({len(self.confirmed_map)} total)")
        self.update_info()

    def save_json(self):
        if not self.confirmed_map:
            messagebox.showinfo("Nothing to save", "No confirmed windows.")
            return
        # Save only confirmed windows, ordered in original list order
        out = {}
        i = 0
        for w in self.windows:
            if w.name in self.confirmed_map:
                payload = self.confirmed_map[w.name]
                key = f"window_{i}"
                out[key] = {
                    "start_ns":   payload["start_ns"],
                    "end_ns":     payload["end_ns"],
                    "duration_s": payload["duration_s"],
                    "single_word_description": payload.get("single_word_description", ""),
                    "index":      payload.get("index", None),
                }
                i += 1

        if not self.json_path:
            messagebox.showerror("Save error", "No source JSON path is known.")
            return

        base_dir  = self.json_path.parent
        base_stem = self.json_path.stem + "_confirmed"
        out_path  = base_dir / f"{base_stem}.json"

        # Auto-increment if exists
        if out_path.exists():
            k = 2
            while True:
                trial = base_dir / f"{base_stem}_v{k}.json"
                if not trial.exists():
                    out_path = trial
                    break
                k += 1

        try:
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            self.status.config(text=f"💾 Saved {i} windows to: {out_path}")
            messagebox.showinfo("Saved", f"Saved {i} windows to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save JSON:\n{e}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
