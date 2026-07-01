#!/usr/bin/env python3
# instance_annotator_sam2.py
# Big image view + SAM-2 "Suggest mask" (box/points) + Accept to JSON + save mask PNGs
# Now includes: Handle / point-of-interaction per item (with optional SAM mask)
#
# Output structure:
#   <image_dir>/<image_stem>/
#     ├── <image_stem>.json
#     ├── <index>.png              # object mask
#     ├── <index>_handle.png       # handle mask (if accepted)
#     └── ...
#
# JSON per item includes:
#   - index, class, description
#   - polygon, bbox, mask_path, prompt
#   - handle: { point, polygon?, bbox?, mask_path?, prompt? }

import json, base64, requests
from pathlib import Path
from typing import List, Dict, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False

SAM_URL = "http://127.0.0.1:8000/segment"  # Updated to correct port

CLASS_OPTIONS = [
    "revolute",
    "prismatic",
]

# Colors
COL_PREVIEW_OBJ = "#00FF88"   # object preview polygon
COL_SAVED_OBJ   = "#56B4E9"   # saved objects outline
COL_HANDLE_PT   = "#FFFFFF"   # handle point cross
COL_PREVIEW_HND = "#FFA500"   # handle preview polygon (orange)
COL_BOX         = "#E69F00"   # drawn box
COL_POLYLINE    = "#FF44FF"   # polyline prompt

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Instance Annotator (SAM2 + Handle)")
        self.geometry("1280x860")

        # Image / view
        self.image_path: Optional[Path] = None
        self.image: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.im_w = 0
        self.im_h = 0

        self.scale = 1.0
        self.offset = [10.0, 10.0]
        self.drag_start = None
        self.is_panning = False

        # Object prompt state
        self.mode = "idle"              # "draw_box", "points_pos", "points_neg", "handle_point", "polyline"
        self.box_start = None
        self._temp_box_end = None
        self.points_pos: List[List[float]] = []
        self.points_neg: List[List[float]] = []
        self.polyline_points: List[List[float]] = []
        self.polyline_closed = False
        self.preview_poly: Optional[List[List[float]]] = None
        self.preview_mask_b64: Optional[str] = None

        # Handle state
        self.handle_point: Optional[List[float]] = None  # [x,y] in image pixels
        self.handle_preview_poly: Optional[List[List[float]]] = None
        self.handle_preview_b64: Optional[str] = None

        # Items / JSON
        self.items: List[Dict] = []
        self.json_path: Optional[Path] = None

        # SAM server state
        self.sam_available = False
        self._sam_btns: List = []  # buttons that require a running SAM server

        # UI
        self._build_ui()
        self._bind_keys()

        if not PIL_OK:
            messagebox.showwarning("Pillow missing", "pip install pillow requests")

        # Check SAM availability after the event loop starts
        self.after(200, self._check_sam)

    # ---------- UI ----------
    def _build_ui(self):
        menubar = tk.Menu(self)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open Image...", command=self.open_image)
        filem.add_command(label="Load JSON", command=self.reload_json)
        filem.add_command(label="Save JSON", command=self.save_json)
        filem.add_separator()
        filem.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filem)
        self.config(menu=menubar)

        main = ttk.Frame(self); main.pack(fill="both", expand=True)

        # Canvas
        left = ttk.Frame(main); left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#1f1f1f")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        # right-click: finish polyline
        self.canvas.bind("<ButtonRelease-3>", self.on_right_click)
        # middle mouse pan or Space+drag
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        # Zoom
        self.canvas.bind("<MouseWheel>", self.on_wheel)      # Windows/macOS
        self.canvas.bind("<Button-4>", self.on_wheel_linux)  # Linux up
        self.canvas.bind("<Button-5>", self.on_wheel_linux)  # Linux down

        # Right panel
        right = ttk.Frame(main, padding=10); right.pack(side="right", fill="y")
        self.lbl_status = ttk.Label(right, text="No image open.")
        self.lbl_status.pack(anchor="w", pady=(0,2))

        frm_sam = ttk.Frame(right); frm_sam.pack(fill="x", pady=(0,6))
        self.lbl_sam = ttk.Label(frm_sam, text="SAM: checking…", foreground="gray")
        self.lbl_sam.pack(side="left")
        ttk.Button(frm_sam, text="↺", width=3, command=self._check_sam).pack(side="left", padx=(6,0))

        # Index / Class / Description
        frm_idx = ttk.Frame(right); frm_idx.pack(fill="x", pady=4)
        ttk.Label(frm_idx, text="Index:").pack(side="left")
        self.var_index = tk.StringVar()
        self.ent_index = ttk.Entry(frm_idx, textvariable=self.var_index, width=8)
        self.ent_index.pack(side="left", padx=(6,0))

        frm_cls = ttk.Frame(right); frm_cls.pack(fill="x", pady=4)
        ttk.Label(frm_cls, text="Class:").pack(side="left")
        self.var_class = tk.StringVar(value=CLASS_OPTIONS[0])
        ttk.OptionMenu(frm_cls, self.var_class, CLASS_OPTIONS[0], *CLASS_OPTIONS).pack(side="left", padx=6)

        frm_desc = ttk.Frame(right); frm_desc.pack(fill="x", pady=4)
        ttk.Label(frm_desc, text="Description:").pack(side="left")
        self.var_desc = tk.StringVar()
        ttk.Entry(frm_desc, textvariable=self.var_desc, width=28).pack(side="left", padx=6)

        # Object prompt controls
        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="Object prompt:").pack(anchor="w")
        ttk.Button(right, text="Draw Box", command=lambda: self.set_mode("draw_box")).pack(fill="x", pady=2)
        ttk.Button(right, text="Point (+)", command=lambda: self.set_mode("points_pos")).pack(fill="x", pady=2)
        ttk.Button(right, text="Point (−)", command=lambda: self.set_mode("points_neg")).pack(fill="x", pady=2)
        ttk.Button(right, text="Clear Obj Prompts", command=self.clear_obj_prompts).pack(fill="x", pady=2)
        ttk.Button(right, text="Polyline", command=lambda: self.set_mode("polyline")).pack(fill="x", pady=2)
        frm_pw = ttk.Frame(right); frm_pw.pack(fill="x", pady=(0,2))
        ttk.Label(frm_pw, text="Polyline stroke (px):").pack(side="left")
        self.var_polyline_width = tk.StringVar(value="10")
        ttk.Entry(frm_pw, textvariable=self.var_polyline_width, width=5).pack(side="left", padx=4)

        ttk.Button(right, text="Suggest Object Mask", command=self.suggest_mask).pack(fill="x", pady=(10,2))

        ttk.Button(right, text="Fill Polygon as Mask", command=self.fill_polygon_as_mask).pack(fill="x", pady=2)
        ttk.Button(right, text="Stroke Polyline as Mask", command=self.fill_polyline_as_mask).pack(fill="x", pady=2)
        ttk.Button(right, text="Accept Object ➜ Save", command=self.accept_mask).pack(fill="x", pady=2)

        # Handle controls
        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="Handle / PoI:").pack(anchor="w")
        frm_h = ttk.Frame(right); frm_h.pack(fill="x", pady=2)
        ttk.Button(frm_h, text="Handle Point", command=lambda: self.set_mode("handle_point")).pack(side="left", fill="x", expand=True, padx=(0,4))
        ttk.Label(right, text="Handle box (px):").pack(anchor="w")
        self.var_handle_box = tk.StringVar(value="80")  # side length in pixels for the local SAM box
        ttk.Entry(right, textvariable=self.var_handle_box, width=8).pack(anchor="w", pady=(0,6))

        btn_suggest_h = ttk.Button(right, text="Suggest Handle Mask (SAM)", command=self.suggest_handle_mask)
        btn_suggest_h.pack(fill="x", pady=2)
        self._sam_btns.append(btn_suggest_h)
        btns = ttk.Frame(right); btns.pack(fill="x")
        ttk.Button(btns, text="Accept Handle ➜ Save", command=self.accept_handle).pack(side="left", fill="x", expand=True, padx=(0,4))
        ttk.Button(btns, text="Clear Handle", command=self.clear_handle).pack(side="left", fill="x", expand=True)

        # Items list
        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="Items:").pack(anchor="w")
        frm_list = ttk.Frame(right); frm_list.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(frm_list, height=12)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_item)
        sb = ttk.Scrollbar(frm_list, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)

        ttk.Button(right, text="Delete Selected", command=self.delete_selected).pack(fill="x", pady=4)
        ttk.Button(right, text="Save JSON", command=self.save_json).pack(fill="x", pady=4)

        # Help
        ttk.Separator(right).pack(fill="x", pady=6)
        ttk.Label(right, text=(
            "Tips:\n"
            "- Space + drag: pan | Mouse wheel: zoom\n"
            "Without SAM (polyline only):\n"
            "  L → click pts → right-click close\n"
            "  → Fill Polygon / Stroke → Accept\n"
            "With SAM:\n"
            "  Box or +/- points → Suggest (SAM) → Accept\n"
            "  Polyline → Suggest (SAM clips to polygon)\n"
            "- Handle: Handle Point → (Suggest SAM) → Accept Handle"
        ), justify="left").pack(anchor="w")

    def _bind_keys(self):
        self.bind("b", lambda e: self.set_mode("draw_box"))
        self.bind("p", lambda e: self.set_mode("points_pos"))
        self.bind("n", lambda e: self.set_mode("points_neg"))
        self.bind("h", lambda e: self.set_mode("handle_point"))
        self.bind("l", lambda e: self.set_mode("polyline"))
        self.bind("=", lambda e: self.zoom(1.1))
        self.bind("-", lambda e: self.zoom(1/1.1))
        self.bind("<space>", lambda e: self.toggle_pan(True))
        self.bind("<KeyRelease-space>", lambda e: self.toggle_pan(False))

    # ---------- Image / JSON paths ----------
    def open_image(self):
        filetypes = [
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(title="Select an Image", filetypes=filetypes)
        if filepath:
            self.image_path = Path(filepath)
            self.image = Image.open(filepath)
            self.photo = ImageTk.PhotoImage(self.image)
            self.im_w, self.im_h = self.image.size
            # Reload any previously saved annotations for this image so we don't
            # have to redo the full annotation of all instances.
            self.load_json()
            self.redraw()
            self.lbl_status.config(
                text=f"Loaded: {self.image_path.name} ({len(self.items)} saved item(s))"
            )
        else:
            messagebox.showinfo("No file selected", "Please select an image file.")

    def out_dir(self) -> Optional[Path]:
        if not self.image_path: return None
        d = self.image_path.parent / "instance_annotations"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def json_default_path(self) -> Optional[Path]:
        d = self.out_dir()
        if not d: return None
        return d / f"{self.image_path.stem}.json"

    def load_json(self):
        self.items.clear()
        self.listbox.delete(0, "end")
        self.json_path = self.json_default_path()
        if self.json_path and self.json_path.exists():
            try:
                data = json.load(open(self.json_path, "r"))
                self.items = data.get("items", [])
                for it in self.items:
                    self.listbox.insert("end", f'{it.get("index")} - {it.get("class")}')
            except Exception as e:
                messagebox.showwarning("JSON load", f"Failed to load JSON: {e}")

    def reload_json(self):
        """Manually reload saved annotations for the current image from disk."""
        if not self.image_path:
            messagebox.showinfo("No image", "Open an image first."); return
        self.load_json()
        self.clear_obj_prompts()
        self.clear_handle()
        self.redraw()
        self.lbl_status.config(text=f"Reloaded {len(self.items)} item(s) from {self.json_path}")

    def save_json(self):
        if not self.image_path:
            messagebox.showinfo("No image", "Open an image first."); return
        out = {
            "image_path": str(self.image_path),
            "image_size": [self.im_w, self.im_h],
            "items": self.items,
        }
        outp = self.json_default_path()
        try:
            json.dump(out, open(outp, "w"), indent=2)
            self.lbl_status.config(text=f"Saved {len(self.items)} items to {outp}")
        except Exception as e:
            messagebox.showerror("Save JSON", str(e))

    # ---------- Canvas helpers ----------
    def img_to_canvas(self, x, y):
        return self.offset[0] + x*self.scale, self.offset[1] + y*self.scale

    def canvas_to_img(self, X, Y):
        return (X - self.offset[0]) / self.scale, (Y - self.offset[1]) / self.scale

    def canvas_to_image(self, x, y):
        """Convert canvas coordinates to image coordinates."""
        img_x = (x - self.offset[0]) / self.scale
        img_y = (y - self.offset[1]) / self.scale
        return img_x, img_y

    def redraw(self):
        self.canvas.delete("all")
        if not self.image: return
        disp = self.image.resize((int(self.im_w*self.scale), int(self.im_h*self.scale)), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.photo)

        # Object box
        if self.box_start and self._temp_box_end:
            x1,y1,x2,y2 = self._current_box()
            X1,Y1 = self.img_to_canvas(x1,y1); X2,Y2 = self.img_to_canvas(x2,y2)
            self.canvas.create_rectangle(X1,Y1,X2,Y2, outline=COL_BOX, width=2)

        # Object points
        for (x,y) in self.points_pos:
            X,Y = self.img_to_canvas(x,y)
            self.canvas.create_oval(X-4,Y-4,X+4,Y+4, outline="#00C853", width=2)
        for (x,y) in self.points_neg:
            X,Y = self.img_to_canvas(x,y)
            self.canvas.create_oval(X-4,Y-4,X+4,Y+4, outline="#D32F2F", width=2)

        # Polyline prompt
        if self.polyline_points:
            canvas_pts = [self.img_to_canvas(x, y) for x, y in self.polyline_points]
            for (X, Y) in canvas_pts:
                self.canvas.create_oval(X-4, Y-4, X+4, Y+4, fill=COL_POLYLINE, outline="")
            if len(canvas_pts) >= 2:
                flat = [v for pt in canvas_pts for v in pt]
                self.canvas.create_line(*flat, fill=COL_POLYLINE, width=2)
            if self.polyline_closed and len(canvas_pts) >= 3:
                X0, Y0 = canvas_pts[0]
                Xl, Yl = canvas_pts[-1]
                self.canvas.create_line(Xl, Yl, X0, Y0, fill=COL_POLYLINE, width=2, dash=(4, 2))

        # Object preview polygon
        if self.preview_poly:
            pts=[]
            for (x,y) in self.preview_poly:
                X,Y = self.img_to_canvas(x,y); pts.extend([X,Y])
            self.canvas.create_polygon(*pts, outline=COL_PREVIEW_OBJ, width=2, fill="", smooth=False)

        # Handle point (draw a little cross)
        if self.handle_point:
            X,Y = self.img_to_canvas(*self.handle_point)
            s = 5
            self.canvas.create_line(X-s, Y, X+s, Y, fill=COL_HANDLE_PT, width=2)
            self.canvas.create_line(X, Y-s, X, Y+s, fill=COL_HANDLE_PT, width=2)

        # Handle preview polygon
        if self.handle_preview_poly:
            pts=[]
            for (x,y) in self.handle_preview_poly:
                X,Y = self.img_to_canvas(x,y); pts.extend([X,Y])
            self.canvas.create_polygon(*pts, outline=COL_PREVIEW_HND, width=2, fill="", dash=(5,2), smooth=True)

        # Saved items outlines
        for it in self.items:
            poly = it.get("polygon")
            if poly:
                pts=[]
                for (x,y) in poly:
                    X,Y = self.img_to_canvas(x,y); pts.extend([X,Y])
                self.canvas.create_polygon(*pts, outline=COL_SAVED_OBJ, width=1, dash=(4,2), fill="")

            # (Optional) draw saved handle point faintly
            h = it.get("handle") or {}
            hp = h.get("point")
            if hp:
                X,Y = self.img_to_canvas(*hp)
                self.canvas.create_oval(X-2,Y-2,X+2,Y+2, outline="#BBBBBB", width=1)

    # ---------- Zoom / pan ----------
    def zoom(self, factor):
        if not self.image: return
        cx = self.canvas.winfo_width()/2; cy = self.canvas.winfo_height()/2
        ix, iy = self.canvas_to_img(cx, cy)
        self.scale *= factor
        nx, ny = self.img_to_canvas(ix, iy)
        self.offset[0] += cx - nx; self.offset[1] += cy - ny
        self.redraw()

    def on_wheel(self, e): self.zoom(1.1 if e.delta > 0 else 1/1.1)
    def on_wheel_linux(self, e): self.zoom(1.1 if e.num == 4 else 1/1.1)

    def toggle_pan(self, enable):
        self.is_panning = enable
        self.canvas.config(cursor="fleur" if enable else "")

    def on_pan_start(self, e):
        self.is_panning = True; self.drag_start = (e.x, e.y); self.canvas.config(cursor="fleur")

    def on_pan_drag(self, e):
        if not self.is_panning or not self.drag_start: return
        dx = e.x - self.drag_start[0]; dy = e.y - self.drag_start[1]
        self.offset[0] += dx; self.offset[1] += dy
        self.drag_start = (e.x, e.y)
        self.redraw()

    def on_pan_end(self, e):
        self.is_panning = False; self.drag_start = None; self.canvas.config(cursor="")

    # ---------- Modes / prompts ----------
    def set_mode(self, m):
        self.mode = m
        self.lbl_status.config(text=f"Mode: {m}")

    def _current_box(self):
        if not self.box_start or not self._temp_box_end: return None
        x1,y1 = self.box_start; x2,y2 = self._temp_box_end
        x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
        x1 = max(0, min(self.im_w-1, x1)); x2 = max(0, min(self.im_w-1, x2))
        y1 = max(0, min(self.im_h-1, y1)); y2 = max(0, min(self.im_h-1, y2))
        return (float(x1), float(y1), float(x2), float(y2))

    def clear_obj_prompts(self):
        self.box_start = None; self._temp_box_end = None
        self.points_pos.clear(); self.points_neg.clear()
        self.polyline_points = []; self.polyline_closed = False
        self.preview_poly = None; self.preview_mask_b64 = None
        self.redraw()

    def clear_handle(self):
        self.handle_point = None
        self.handle_preview_poly = None
        self.handle_preview_b64 = None
        self.redraw()

    # ---------- Mouse ----------
    def on_left_down(self, e):
        if not self.image: return
        x,y = self.canvas_to_img(e.x, e.y)
        if self.is_panning:
            self.drag_start = (e.x, e.y); return

        if self.mode == "draw_box":
            self.box_start = (x,y); self._temp_box_end = (x,y)

        elif self.mode == "points_pos":
            self.points_pos.append([float(x), float(y)])

        elif self.mode == "points_neg":
            self.points_neg.append([float(x), float(y)])

        elif self.mode == "handle_point":
            self.handle_point = [float(max(0, min(self.im_w-1, x))),
                                 float(max(0, min(self.im_h-1, y)))]
            # Clear previous handle preview (user will suggest new mask)
            self.handle_preview_poly = None
            self.handle_preview_b64 = None

        elif self.mode == "polyline":
            if not self.polyline_closed:
                self.polyline_points.append([float(x), float(y)])

        self.redraw()

    def on_left_drag(self, e):
        if not self.image: return
        if self.is_panning and self.drag_start:
            dx = e.x - self.drag_start[0]; dy = e.y - self.drag_start[1]
            self.offset[0] += dx; self.offset[1] += dy
            self.drag_start = (e.x, e.y)
            self.redraw(); return
        if self.mode == "draw_box" and self.box_start:
            x2,y2 = self.canvas_to_img(e.x, e.y)
            self._temp_box_end = (x2,y2); self.redraw()

    def on_left_up(self, e):
        if not self.image: return
        if self.is_panning:
            self.drag_start=None; return
        if self.mode == "draw_box" and self.box_start:
            x2,y2 = self.canvas_to_img(e.x, e.y)
            self._temp_box_end = (x2,y2); self.redraw()

    def on_right_click(self, e):
        """Close the current polyline (right-click to finish)."""
        if self.mode == "polyline" and len(self.polyline_points) >= 3:
            self.polyline_closed = True
            self.lbl_status.config(text=f"Polyline closed ({len(self.polyline_points)} pts). Click 'Suggest Object Mask'.")
            self.redraw()

    # ---------- SAM availability ----------
    def _check_sam(self):
        health_url = SAM_URL.replace("/segment", "/health")
        try:
            r = requests.get(health_url, timeout=2)
            self.sam_available = r.status_code == 200
        except Exception:
            self.sam_available = False
        self._update_sam_ui()

    def _update_sam_ui(self):
        if self.sam_available:
            self.lbl_sam.config(text=f"SAM: online ({SAM_URL})", foreground="green")
        else:
            self.lbl_sam.config(text="SAM: offline — polyline only", foreground="orange")
        state = "normal" if self.sam_available else "disabled"
        for btn in self._sam_btns:
            btn.config(state=state)

    # ---------- SAM requests ----------
    def _clip_mask_to_polyline(self, mask_b64: str) -> str:
        """Intersect a base64 mask PNG with the filled polyline polygon."""
        from PIL import ImageDraw, ImageChops
        import io
        mask_bytes = base64.b64decode(mask_b64.encode("ascii"))
        sam_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        poly_mask = Image.new("L", (self.im_w, self.im_h), 0)
        ImageDraw.Draw(poly_mask).polygon(
            [(x, y) for x, y in self.polyline_points], fill=255
        )

        clipped = ImageChops.multiply(sam_mask, poly_mask)
        buf = io.BytesIO()
        clipped.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def suggest_mask(self):
        """Object mask from current object prompts. Falls back to polygon fill if SAM is offline."""
        if not self.image_path:
            messagebox.showinfo("No image", "Open an image first."); return

        if self.polyline_points:
            # Without SAM: fill the closed polygon directly (no server needed)
            if not self.sam_available:
                self.fill_polygon_as_mask()
                return
            # With SAM: use bbox of polyline, then clip result to polygon
            xs = [p[0] for p in self.polyline_points]
            ys = [p[1] for p in self.polyline_points]
            bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
            try:
                r = requests.post(SAM_URL, json={
                    "image_path": str(self.image_path),
                    "bbox": bbox,
                    "polyline": self.polyline_points,
                }, timeout=20)
                r.raise_for_status()
                data = r.json()
                if not data.get("ok") or not data.get("polygons"):
                    self.lbl_status.config(text="SAM2: no object mask found.")
                    self.preview_poly = None; self.preview_mask_b64 = None
                else:
                    clipped_b64 = self._clip_mask_to_polyline(data["mask_png_b64"])
                    self.preview_poly = list(self.polyline_points)
                    self.preview_mask_b64 = clipped_b64
                    self.lbl_status.config(text=f"SAM2: mask clipped to polyline ({len(self.polyline_points)} pts)")
                self.redraw()
            except Exception as e:
                messagebox.showerror("SAM2 error", str(e))
        else:
            bbox = self._current_box()
            pts = self.points_pos + self.points_neg
            if not bbox and not pts:
                messagebox.showinfo("Need prompt", "Draw a box, add points (+/−), or draw a polyline first."); return
            if not self.sam_available:
                messagebox.showinfo("SAM offline", "SAM server is not running.\nUse the Polyline tool (L key) to annotate without SAM."); return
            labels = [1]*len(self.points_pos) + [0]*len(self.points_neg)
            try:
                r = requests.post(SAM_URL, json={
                    "image_path": str(self.image_path),
                    "bbox": bbox, "points": pts, "labels": labels
                }, timeout=20)
                r.raise_for_status()
                data = r.json()
                if not data.get("ok") or not data.get("polygons"):
                    self.lbl_status.config(text="SAM2: no object mask found.")
                    self.preview_poly = None; self.preview_mask_b64 = None
                else:
                    self.preview_poly = data["polygons"][0]
                    self.preview_mask_b64 = data.get("mask_png_b64", "")
                    self.lbl_status.config(text=f"SAM2: object mask preview ({len(self.preview_poly)} pts)")
                self.redraw()
            except Exception as e:
                messagebox.showerror("SAM2 error", str(e))

    def fill_polyline_as_mask(self):
        """Stroke the polyline path directly as the object mask (no SAM)."""
        if not self.image_path:
            messagebox.showinfo("No image", "Open an image first."); return
        if len(self.polyline_points) < 2:
            messagebox.showinfo("Polyline", "Draw a polyline with at least 2 points first."); return

        from PIL import ImageDraw
        import io

        try:
            width = max(1, int(self.var_polyline_width.get()))
        except Exception:
            width = 10

        mask_img = Image.new("L", (self.im_w, self.im_h), 0)
        draw = ImageDraw.Draw(mask_img)

        pts = [(x, y) for x, y in self.polyline_points]
        if self.polyline_closed:
            pts = pts + [pts[0]]  # close the loop

        draw.line(pts, fill=255, width=width)

        # round caps at each vertex to fill gaps at joints
        r = width / 2.0
        for (x, y) in pts:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        self.preview_mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        self.preview_poly = list(self.polyline_points)
        self.lbl_status.config(text=f"Polyline stroked as mask ({len(self.polyline_points)} pts, {width}px). Accept to save.")
        self.redraw()

    def fill_polygon_as_mask(self):
        """Fill the closed polyline polygon as the object mask (no SAM needed)."""
        if not self.image_path:
            messagebox.showinfo("No image", "Open an image first."); return
        if len(self.polyline_points) < 3:
            messagebox.showinfo("Polyline", "Draw a polyline with at least 3 points first."); return
        if not self.polyline_closed:
            messagebox.showinfo("Polyline", "Right-click to close the polyline first."); return

        from PIL import ImageDraw
        import io

        mask_img = Image.new("L", (self.im_w, self.im_h), 0)
        ImageDraw.Draw(mask_img).polygon(
            [(x, y) for x, y in self.polyline_points], fill=255
        )

        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        self.preview_mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        self.preview_poly = list(self.polyline_points)
        self.lbl_status.config(text=f"Polygon filled as mask ({len(self.polyline_points)} pts). Accept to save.")
        self.redraw()

    def suggest_handle_mask(self):
        """Handle mask from a small box around the handle point (plus the point itself)."""
        if not self.image_path:
            messagebox.showinfo("No image", "Open an image first."); return
        if not self.handle_point:
            messagebox.showinfo("Handle", "Set a handle point first (click 'Handle Point')."); return
        # small local box around handle
        try:
            size = max(8, int(self.var_handle_box.get()))
        except Exception:
            size = 80
        cx, cy = self.handle_point
        half = size / 2.0
        x1 = max(0.0, cx - half); y1 = max(0.0, cy - half)
        x2 = min(self.im_w-1.0, cx + half); y2 = min(self.im_h-1.0, cy + half)
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        pts  = [self.handle_point]  # positive point at handle
        labels = [1]
        try:
            r = requests.post(SAM_URL, json={
                "image_path": str(self.image_path),
                "bbox": bbox, "points": pts, "labels": labels
            }, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok") or not data.get("polygons"):
                self.lbl_status.config(text="SAM2: no handle mask found.")
                self.handle_preview_poly = None; self.handle_preview_b64 = None
            else:
                self.handle_preview_poly = data["polygons"][0]
                self.handle_preview_b64  = data.get("mask_png_b64", "")
                self.lbl_status.config(text=f"SAM2: handle mask preview ({len(self.handle_preview_poly)} pts)")
            self.redraw()
        except Exception as e:
            messagebox.showerror("SAM2 error", str(e))

    # ---------- Accept / Save ----------
    def accept_mask(self):
        """Accept object mask preview -> save main entry (upsert)."""
        if not self.preview_poly or not self.preview_mask_b64:
            messagebox.showinfo("No mask", "Get a SAM2 object mask preview first."); return

        idx_txt = (self.var_index.get() or "").strip()
        if not idx_txt.isdigit():
            messagebox.showerror("Index", "Index must be an integer."); return
        idx_val = int(idx_txt)

        out_dir = self.out_dir()
        if not out_dir:
            messagebox.showerror("Output", "Cannot resolve output directory."); return

        # Save object mask PNG
        mask_bytes = base64.b64decode(self.preview_mask_b64.encode("ascii"))
        mask_path = out_dir / f"{idx_val}.png"
        try:
            with open(mask_path, "wb") as f:
                f.write(mask_bytes)
        except Exception as e:
            messagebox.showerror("Write mask", str(e)); return

        payload = {
            "index": idx_val,
            "class": self.var_class.get().strip(),
            "description": self.var_desc.get().strip(),
            "polygon": [[int(round(x)), int(round(y))] for (x,y) in self.preview_poly],
            "mask_path": str(mask_path),
            "prompt": {
                "points_pos": self.points_pos,
                "points_neg": self.points_neg,
                "bbox": self._current_box(),
                "polyline": list(self.polyline_points) if self.polyline_points else None,
            }
        }
        # include bbox
        if self._current_box():
            x1,y1,x2,y2 = self._current_box()
            payload["bbox"] = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

        # Upsert by index
        existing = None
        for it in self.items:
            if it.get("index") == idx_val:
                existing = it; break
        if existing:
            existing.update(payload)
            i = self.items.index(existing)
            self.listbox.delete(i); self.listbox.insert(i, f'{payload["index"]} - {payload["class"]}')
        else:
            self.items.append(payload)
            self.listbox.insert("end", f'{payload["index"]} - {payload["class"]}')

        self.save_json()
        self.lbl_status.config(text=f"Saved OBJECT for index {idx_val}")
        # keep preview or clear prompts if you prefer
        # self.clear_obj_prompts()

    def accept_handle(self):
        """Accept handle (point + optional preview mask)."""
        idx_txt = (self.var_index.get() or "").strip()
        if not idx_txt.isdigit():
            messagebox.showerror("Index", "Index must be an integer."); return
        idx_val = int(idx_txt)

        if not self.handle_point:
            messagebox.showinfo("Handle", "Set a handle point first."); return

        out_dir = self.out_dir()
        if not out_dir:
            messagebox.showerror("Output", "Cannot resolve output directory."); return

        # Build handle payload
        handle = {
            "point": [int(round(self.handle_point[0])), int(round(self.handle_point[1]))],
            "polygon": None,
            "bbox": None,
            "mask_path": None,
            "prompt": None,
        }

        # If we have a handle mask preview, save it and add polygon+box
        if self.handle_preview_b64 and self.handle_preview_poly:
            mask_bytes = base64.b64decode(self.handle_preview_b64.encode("ascii"))
            mask_path = out_dir / f"{idx_val}_handle.png"
            try:
                with open(mask_path, "wb") as f:
                    f.write(mask_bytes)
                handle["mask_path"] = str(mask_path)
            except Exception as e:
                messagebox.showerror("Write handle mask", str(e)); return

            # bbox (AABB) of handle polygon
            xs = [p[0] for p in self.handle_preview_poly]
            ys = [p[1] for p in self.handle_preview_poly]
            handle["polygon"] = [[int(round(x)), int(round(y))] for (x,y) in self.handle_preview_poly]
            handle["bbox"] = [int(round(min(xs))), int(round(min(ys))),
                              int(round(max(xs))), int(round(max(ys)))]

            # prompt info we used
            try:
                size = max(8, int(self.var_handle_box.get()))
            except Exception:
                size = 80
            cx, cy = self.handle_point
            half = size/2.0
            x1 = max(0.0, cx-half); y1 = max(0.0, cy-half)
            x2 = min(self.im_w-1.0, cx+half); y2 = min(self.im_h-1.0, cy+half)
            handle["prompt"] = {
                "point": [self.handle_point[0], self.handle_point[1]],
                "bbox": [x1,y1,x2,y2]
            }

        # Upsert into item
        existing = None
        for it in self.items:
            if it.get("index") == idx_val:
                existing = it; break

        if existing:
            existing.setdefault("handle", {}).update(handle)
            i = self.items.index(existing)
            # update label
            self.listbox.delete(i); self.listbox.insert(i, f'{existing["index"]} - {existing.get("class","other")}')
        else:
            # create a minimal item with just handle (and info fields)
            payload = {
                "index": idx_val,
                "class": self.var_class.get().strip(),
                "description": self.var_desc.get().strip(),
                "handle": handle
            }
            self.items.append(payload)
            self.listbox.insert("end", f'{payload["index"]} - {payload["class"]}')

        self.save_json()
        self.lbl_status.config(text=f"Saved HANDLE for index {idx_val}")

    # ---------- List selection / delete ----------
    def on_select_item(self, e):
        sel = self.listbox.curselection()
        if not sel: return
        i = sel[0]; item = self.items[i]
        self.var_index.set(str(item.get("index","")))
        self.var_class.set(item.get("class","other"))
        self.var_desc.set(item.get("description",""))

        # Load object preview — reload mask PNG from disk so Accept re-save works
        self.preview_poly = item.get("polygon")
        mask_path = item.get("mask_path")
        if mask_path and Path(mask_path).exists():
            try:
                with open(mask_path, "rb") as f:
                    self.preview_mask_b64 = base64.b64encode(f.read()).decode("ascii")
            except Exception:
                self.preview_mask_b64 = None
        else:
            self.preview_mask_b64 = None

        # Restore object bbox into draw box UI (if present)
        if "bbox" in item and item["bbox"]:
            x1,y1,x2,y2 = item["bbox"]
            self.box_start = (x1,y1); self._temp_box_end = (x2,y2)
        else:
            self.box_start = None; self._temp_box_end = None

        # Restore polyline from saved prompt (if present)
        prompt = item.get("prompt") or {}
        saved_poly = prompt.get("polyline")
        if saved_poly:
            self.polyline_points = list(saved_poly)
            self.polyline_closed = True
        else:
            self.polyline_points = []; self.polyline_closed = False

        # Restore handle
        h = item.get("handle") or {}
        self.handle_point = h.get("point")
        self.handle_preview_poly = h.get("polygon")  # show saved outline faintly in preview color
        self.handle_preview_b64  = None  # not stored; only during preview

        self.redraw()

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel: return
        i = sel[0]; item = self.items[i]
        # optionally delete mask files on disk
        for k in ("mask_path",):
            try:
                mp = item.get(k)
                if mp and Path(mp).exists(): Path(mp).unlink()
            except Exception:
                pass
        # delete handle mask too
        try:
            h = item.get("handle") or {}
            mp = h.get("mask_path")
            if mp and Path(mp).exists(): Path(mp).unlink()
        except Exception:
            pass

        del self.items[i]; self.listbox.delete(i)
        self.save_json()
        self.preview_poly = None
        self.clear_handle()
        self.redraw()


if __name__ == "__main__":
    if not PIL_OK:
        print("Please: pip install pillow requests")
    App().mainloop()
