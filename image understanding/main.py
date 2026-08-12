import os
import time
import warnings
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

warnings.filterwarnings('ignore')

from cbir.dataset import load_dataset
from cbir.descriptors import color_histogram, lbp_histogram, get_glcm_features
from cbir.search import retrieve_combined

DATASET_DIR = r"C:\Users\User\Downloads\cifar-10-python\cifar-10-batches-py"
CACHE_FILE   = "cbir_cache.npz"
QUERY_PANEL_N = 24
IMG_THUMB    = 48
IMG_RESULT   = 64
COLS_RESULTS = 5

def numpy_to_photoimage(arr, size=None):
    img = Image.fromarray(arr)
    if size:
        img = img.resize((size, size), Image.NEAREST)
    return ImageTk.PhotoImage(img)

def add_border(arr, color, border=4, size=64):
    img = Image.fromarray(arr).resize((size - border * 2, size - border * 2), Image.NEAREST)
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(img, (border, border))
    return ImageTk.PhotoImage(canvas)

class CBIRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CIFAR-10 Content-Based Image Retrieval")
        self.configure(bg="#f0f2f5")
        self.resizable(True, True)
        self.minsize(1100, 700)

        self.db_images = self.db_labels = self.db_filenames = None
        self.test_images = self.test_labels = self.test_filenames = None
        self.label_names = None
        self.db_color = self.db_lbp = self.db_glcm = None

        self.query_indices   = []
        self.selected_q_idx  = None
        self._query_photos   = []
        self._query_frames   = []
        self._result_photos  = []

        self._build_ui()
        self.after(100, self._start_loading)

    def _build_ui(self):
        header = tk.Frame(self, bg="#1e293b", pady=10)
        header.pack(fill=tk.X)
        tk.Label(header,
                 text="Content-Based Image Retrieval  ·  CIFAR-10",
                 font=("Segoe UI", 16, "bold"),
                 bg="#1e293b", fg="white").pack()
        tk.Label(header,
                 text="Color Histogram  |  Local Binary Pattern (LBP)  |  GLCM Texture Features",
                 font=("Segoe UI", 9),
                 bg="#1e293b", fg="#94a3b8").pack()

        self.status_var = tk.StringVar(value="Initialising…")
        self._status_bar = tk.Label(self, textvariable=self.status_var,
                                    font=("Segoe UI", 9), bg="#334155", fg="#e2e8f0",
                                    anchor="w", padx=8, pady=4)
        self._status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill=tk.X, side=tk.BOTTOM)

        body = tk.Frame(self, bg="#f0f2f5")
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self._build_left_panel(body)
        self._build_right_panel(body)

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg="#f0f2f5", width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        settings_card = self._card(left, "⚙  Search Settings")

        tk.Label(settings_card, text="Distance Metric:", bg="white",
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=4)
        self.metric_var = tk.StringVar(value="L2")
        metric_frame = tk.Frame(settings_card, bg="white")
        metric_frame.grid(row=0, column=1, sticky="w", padx=8)
        tk.Radiobutton(metric_frame, text="L2 (Euclidean)", variable=self.metric_var,
                       value="L2", bg="white", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(metric_frame, text="L1 (Manhattan)", variable=self.metric_var,
                       value="L1", bg="white", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=8)

        tk.Label(settings_card, text="Top-K Results:", bg="white",
                 font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", pady=4)
        k_row = tk.Frame(settings_card, bg="white")
        k_row.grid(row=1, column=1, sticky="ew", padx=8)
        self.k_var = tk.IntVar(value=10)
        self.k_label = tk.Label(k_row, textvariable=self.k_var,
                                bg="white", font=("Segoe UI", 9, "bold"), width=3)
        self.k_label.pack(side=tk.RIGHT)
        ttk.Scale(k_row, from_=1, to=30, variable=self.k_var,
                  orient=tk.HORIZONTAL, length=200,
                  command=lambda v: self.k_var.set(int(float(v)))).pack(side=tk.LEFT)

        sep = tk.Frame(settings_card, bg="#e2e8f0", height=1)
        sep.grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)
        tk.Label(settings_card, text="Feature Weights:", bg="white",
                 font=("Segoe UI", 9, "bold")).grid(row=3, column=0, columnspan=2, sticky="w")

        self.w_color = tk.DoubleVar(value=33)
        self.w_lbp   = tk.DoubleVar(value=33)
        self.w_glcm  = tk.DoubleVar(value=34)

        self._weight_row(settings_card, 4, "Color Histogram:", self.w_color, "#2563eb")
        self._weight_row(settings_card, 5, "LBP Texture:",     self.w_lbp,   "#7c3aed")
        self._weight_row(settings_card, 6, "GLCM Features:",   self.w_glcm,  "#0891b2")

        tk.Button(settings_card, text="Equalize Weights", bg="#e2e8f0",
                  font=("Segoe UI", 8), relief=tk.FLAT, cursor="hand2",
                  command=self._equalize_weights).grid(row=7, column=0, columnspan=2,
                                                       sticky="ew", pady=(8, 0))
        settings_card.columnconfigure(1, weight=1)

        q_header = tk.Frame(parent.master if False else left, bg="#f0f2f5")
        q_card_wrapper = tk.Frame(left, bg="#f0f2f5")
        q_card_wrapper.pack(fill=tk.BOTH, expand=True)

        q_title_row = tk.Frame(q_card_wrapper, bg="white",
                               relief=tk.FLAT, bd=0)
        q_title_row.pack(fill=tk.X)
        tk.Label(q_title_row, text="🖼  Select Query Image",
                 font=("Segoe UI", 10, "bold"), bg="white", pady=8, padx=10).pack(side=tk.LEFT)
        tk.Button(q_title_row, text="↻  New Set", bg="#2563eb", fg="white",
                  font=("Segoe UI", 8, "bold"), relief=tk.FLAT, cursor="hand2", padx=8,
                  command=self._load_query_panel).pack(side=tk.RIGHT, padx=6, pady=4)

        q_scroll_frame = tk.Frame(q_card_wrapper, bg="white",
                                  relief=tk.GROOVE, bd=1)
        q_scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.query_canvas = tk.Canvas(q_scroll_frame, bg="white",
                                      highlightthickness=0)
        q_scrollbar = ttk.Scrollbar(q_scroll_frame, orient="vertical",
                                    command=self.query_canvas.yview)
        self.query_canvas.configure(yscrollcommand=q_scrollbar.set)
        q_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.query_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.query_inner = tk.Frame(self.query_canvas, bg="white")
        self.query_canvas_window = self.query_canvas.create_window(
            (0, 0), window=self.query_inner, anchor="nw")
        self.query_inner.bind("<Configure>", lambda e: self.query_canvas.configure(
            scrollregion=self.query_canvas.bbox("all")))
        self.query_canvas.bind("<Configure>", lambda e: self.query_canvas.itemconfig(
            self.query_canvas_window, width=e.width))

    def _weight_row(self, parent, row, label, var, color):
        tk.Label(parent, text=label, bg="white",
                 font=("Segoe UI", 8)).grid(row=row, column=0, sticky="w", pady=2)
        row_frame = tk.Frame(parent, bg="white")
        row_frame.grid(row=row, column=1, sticky="ew", padx=8)
        lbl = tk.Label(row_frame, bg=color, fg="white",
                       font=("Segoe UI", 8, "bold"), width=4)
        lbl.pack(side=tk.RIGHT, padx=(4, 0))
        def _update(v, lbl=lbl, var=var):
            lbl.configure(text=f"{int(float(v))}")
        ttk.Scale(row_frame, from_=0, to=100, variable=var,
                  orient=tk.HORIZONTAL, length=160,
                  command=_update).pack(side=tk.LEFT)
        _update(var.get())

    def _equalize_weights(self):
        self.w_color.set(33)
        self.w_lbp.set(33)
        self.w_glcm.set(34)

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg="#f0f2f5")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        metrics_frame = tk.Frame(right, bg="#1e293b", pady=6)
        metrics_frame.pack(fill=tk.X)
        self.precision_var = tk.StringVar(value="Precision@K: —")
        self.latency_var   = tk.StringVar(value="Search time: —")
        tk.Label(metrics_frame, textvariable=self.precision_var,
                 font=("Segoe UI", 10, "bold"), bg="#1e293b", fg="#10b981").pack(side=tk.LEFT, padx=16)
        tk.Label(metrics_frame, textvariable=self.latency_var,
                 font=("Segoe UI", 9), bg="#1e293b", fg="#94a3b8").pack(side=tk.LEFT)

        leg = tk.Frame(right, bg="white", pady=4)
        leg.pack(fill=tk.X)
        tk.Label(leg, text="■", fg="#10b981", bg="white",
                 font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(12, 2))
        tk.Label(leg, text="Correct class match", bg="white",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Label(leg, text="■", fg="#ef4444", bg="white",
                 font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(16, 2))
        tk.Label(leg, text="Different class", bg="white",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)

        results_outer = tk.Frame(right, bg="#f0f2f5")
        results_outer.pack(fill=tk.BOTH, expand=True, pady=4)

        self.results_canvas = tk.Canvas(results_outer, bg="#f0f2f5",
                                        highlightthickness=0)
        r_scroll = ttk.Scrollbar(results_outer, orient="vertical",
                                 command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=r_scroll.set)
        r_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.results_inner = tk.Frame(self.results_canvas, bg="#f0f2f5")
        self.results_canvas_window = self.results_canvas.create_window(
            (0, 0), window=self.results_inner, anchor="nw")
        self.results_inner.bind("<Configure>", lambda e: self.results_canvas.configure(
            scrollregion=self.results_canvas.bbox("all")))
        self.results_canvas.bind("<Configure>", lambda e: self.results_canvas.itemconfig(
            self.results_canvas_window, width=e.width))

        self._show_placeholder()

    def _card(self, parent, title):
        wrapper = tk.Frame(parent, bg="white", relief=tk.FLAT, bd=1,
                           highlightbackground="#e2e8f0", highlightthickness=1)
        wrapper.pack(fill=tk.X, pady=(0, 8))
        tk.Label(wrapper, text=title, font=("Segoe UI", 10, "bold"),
                 bg="white", pady=8, padx=10).pack(anchor="w")
        sep = tk.Frame(wrapper, bg="#e2e8f0", height=1)
        sep.pack(fill=tk.X)
        inner = tk.Frame(wrapper, bg="white", padx=10, pady=10)
        inner.pack(fill=tk.X)
        return inner

    def _start_loading(self):
        self._progress.start(10)
        self.status_var.set("Loading CIFAR-10 dataset and features…")
        t = threading.Thread(target=self._load_data, daemon=True)
        t.start()

    def _load_data(self):
        try:
            (self.db_images, self.db_labels, self.db_filenames,
             self.test_images, self.test_labels, self.test_filenames,
             self.label_names) = load_dataset(DATASET_DIR)

            if os.path.exists(CACHE_FILE):
                self.after(0, lambda: self.status_var.set("Loading cached features…"))
                cache = np.load(CACHE_FILE)
                self.db_color = cache['db_color']
                self.db_lbp   = cache['db_lbp']
                self.db_glcm  = cache['db_glcm']
            else:
                self.after(0, lambda: self.status_var.set(
                    "Extracting features (one-time, ~30 s)…"))
                self.db_color = np.array([color_histogram(img) for img in self.db_images],
                                         dtype=np.float32)
                self.db_lbp   = np.array([lbp_histogram(img)  for img in self.db_images],
                                         dtype=np.float32)
                self.db_glcm  = np.array([get_glcm_features(img) for img in self.db_images],
                                         dtype=np.float32)
                np.savez_compressed(CACHE_FILE,
                                    db_color=self.db_color,
                                    db_lbp=self.db_lbp,
                                    db_glcm=self.db_glcm)

            self.after(0, self._on_loaded)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Load Error", str(e)))
            self.after(0, self._progress.stop)

    def _on_loaded(self):
        self._progress.stop()
        self._progress.pack_forget()
        n = len(self.db_images)
        self.status_var.set(
            f"Ready  ·  Database: {n:,} images  ·  Test set: {len(self.test_images):,} images"
            f"  ·  Classes: {', '.join(self.label_names)}")
        self._load_query_panel()

    def _load_query_panel(self):
        if self.test_images is None:
            return
        for w in self.query_inner.winfo_children():
            w.destroy()
        self._query_photos.clear()
        self._query_frames.clear()
        self.selected_q_idx = None

        self.query_indices = list(
            np.random.choice(len(self.test_images), QUERY_PANEL_N, replace=False))

        COLS = 4
        for pos, idx in enumerate(self.query_indices):
            img_arr = self.test_images[idx]
            photo = numpy_to_photoimage(img_arr, IMG_THUMB)
            self._query_photos.append(photo)

            row, col = divmod(pos, COLS)
            frame = tk.Frame(self.query_inner, bg="white",
                             relief=tk.FLAT, cursor="hand2",
                             highlightbackground="#e2e8f0", highlightthickness=1)
            frame.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            self._query_frames.append(frame)

            img_lbl = tk.Label(frame, image=photo, bg="white")
            img_lbl.pack()
            cls_lbl = tk.Label(frame, text=self.label_names[self.test_labels[idx]],
                               font=("Segoe UI", 7), bg="white", fg="#64748b")
            cls_lbl.pack()

            for widget in (frame, img_lbl, cls_lbl):
                widget.bind("<Button-1>",
                            lambda e, i=idx, f=frame: self._select_query(i, f))

        for c in range(COLS):
            self.query_inner.columnconfigure(c, weight=1)

    def _select_query(self, idx, frame):
        for f in self._query_frames:
            f.configure(highlightbackground="#e2e8f0", highlightthickness=1)
        frame.configure(highlightbackground="#2563eb", highlightthickness=2)
        self.selected_q_idx = idx
        self._run_search(idx)

    def _run_search(self, q_idx):
        self.status_var.set(f"Searching for similar images to test[{q_idx}]…")
        t = threading.Thread(target=self._search_thread, args=(q_idx,), daemon=True)
        t.start()

    def _search_thread(self, q_idx):
        q_img   = self.test_images[q_idx]
        q_label = self.test_labels[q_idx]

        t0 = time.time()
        q_color = color_histogram(q_img)
        q_lbp   = lbp_histogram(q_img)
        q_glcm  = get_glcm_features(q_img)

        top_k  = self.k_var.get()
        metric = self.metric_var.get()
        wc = self.w_color.get()
        wl = self.w_lbp.get()
        wg = self.w_glcm.get()

        indices, combined_dists, color_dists, lbp_dists, glcm_dists = retrieve_combined(
            q_color, q_lbp, q_glcm,
            self.db_color, self.db_lbp, self.db_glcm,
            w_color=wc, w_lbp=wl, w_glcm=wg,
            metric=metric, top_k=top_k
        )
        elapsed_ms = (time.time() - t0) * 1000

        results = []
        for rank in range(top_k):
            mi = int(indices[rank])
            results.append({
                "rank": rank + 1,
                "index": mi,
                "img": self.db_images[mi],
                "label": int(self.db_labels[mi]),
                "correct": bool(self.db_labels[mi] == q_label),
                "dist": float(combined_dists[rank]),
                "dist_color": float(color_dists[rank]),
                "dist_lbp": float(lbp_dists[rank]),
                "dist_glcm": float(glcm_dists[rank]),
            })

        precision = sum(1 for r in results if r["correct"]) / top_k
        self.after(0, lambda: self._show_results(q_img, q_label, q_idx,
                                                  results, precision, elapsed_ms))

    def _show_placeholder(self):
        for w in self.results_inner.winfo_children():
            w.destroy()
        self._result_photos.clear()
        ph = tk.Frame(self.results_inner, bg="#f0f2f5")
        ph.pack(expand=True, fill=tk.BOTH, pady=80)
        tk.Label(ph, text="🔍", font=("Segoe UI", 36), bg="#f0f2f5").pack()
        tk.Label(ph, text="Select a query image from the left panel",
                 font=("Segoe UI", 11), bg="#f0f2f5", fg="#94a3b8").pack(pady=8)

    def _show_results(self, q_img, q_label, q_idx, results, precision, elapsed_ms):
        self.precision_var.set(f"Precision@{len(results)}: {precision:.0%}")
        self.latency_var.set(f"Search time: {elapsed_ms:.1f} ms")

        for w in self.results_inner.winfo_children():
            w.destroy()
        self._result_photos.clear()

        q_frame = tk.Frame(self.results_inner, bg="white",
                           relief=tk.FLAT, padx=12, pady=8,
                           highlightbackground="#2563eb", highlightthickness=2)
        q_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        q_photo = add_border(q_img, "#2563eb", border=3, size=80)
        self._result_photos.append(q_photo)
        tk.Label(q_frame, image=q_photo, bg="white").pack(side=tk.LEFT, padx=(0, 12))

        info_frame = tk.Frame(q_frame, bg="white")
        info_frame.pack(side=tk.LEFT, anchor="w")
        tk.Label(info_frame, text="QUERY IMAGE",
                 font=("Segoe UI", 8), bg="white", fg="#64748b").pack(anchor="w")
        tk.Label(info_frame, text=f"Class: {self.label_names[q_label].upper()}",
                 font=("Segoe UI", 13, "bold"), bg="white", fg="#1e293b").pack(anchor="w")
        tk.Label(info_frame, text=f"Test index: {q_idx}",
                 font=("Segoe UI", 9), bg="white", fg="#64748b").pack(anchor="w")
        tk.Label(info_frame,
                 text=f"Metric: {self.metric_var.get()}  ·  K={len(results)}  ·  "
                      f"W(color={self.w_color.get():.0f}, lbp={self.w_lbp.get():.0f}, glcm={self.w_glcm.get():.0f})",
                 font=("Segoe UI", 8), bg="white", fg="#94a3b8").pack(anchor="w")

        grid_label = tk.Frame(self.results_inner, bg="#f0f2f5")
        grid_label.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(grid_label,
                 text=f"Top-{len(results)} Retrieved Images  (sorted by similarity, closest first)",
                 font=("Segoe UI", 9, "bold"), bg="#f0f2f5", fg="#475569").pack(side=tk.LEFT)

        grid = tk.Frame(self.results_inner, bg="#f0f2f5")
        grid.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        for i, r in enumerate(results):
            row, col = divmod(i, COLS_RESULTS)
            color = "#10b981" if r["correct"] else "#ef4444"

            cell = tk.Frame(grid, bg="white",
                            highlightbackground=color, highlightthickness=2)
            cell.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

            photo = add_border(r["img"], color, border=3, size=IMG_RESULT)
            self._result_photos.append(photo)

            tk.Label(cell, image=photo, bg="white").pack(pady=(6, 2))

            cls_name = self.label_names[r["label"]]
            cls_color = "#10b981" if r["correct"] else "#ef4444"
            tk.Label(cell, text=f"#{r['rank']}  {cls_name}",
                     font=("Segoe UI", 8, "bold"), bg="white", fg=cls_color).pack()
            tk.Label(cell, text=f"dist: {r['dist']:.4f}",
                     font=("Segoe UI", 7), bg="white", fg="#94a3b8").pack(pady=(0, 2))

            sub = tk.Frame(cell, bg="white")
            sub.pack(fill=tk.X, padx=6, pady=(0, 6))
            for tag, val, clr in [
                ("C", r["dist_color"], "#2563eb"),
                ("L", r["dist_lbp"],   "#7c3aed"),
                ("G", r["dist_glcm"],  "#0891b2"),
            ]:
                tk.Label(sub, text=f"{tag}:{val:.3f}",
                         font=("Courier", 7), bg="white", fg=clr).pack(side=tk.LEFT, padx=1)

        for c in range(COLS_RESULTS):
            grid.columnconfigure(c, weight=1)

        self.status_var.set(
            f"Query: test[{q_idx}] = {self.label_names[q_label]}  ·  "
            f"Precision@{len(results)}: {precision:.0%}  ·  {elapsed_ms:.1f} ms")

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("ERROR: Pillow is required.  Run:  python -m pip install Pillow")
        raise SystemExit(1)

    app = CBIRApp()
    app.mainloop()
