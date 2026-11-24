"""
Qo'lyozma harfni aniqlash — sinov dasturi.
Sichqoncha bilan harf chizing, model uni A–Z harflaridan biri sifatida aniqlaydi.

Ishga tushirish:  python harf_chizish.py
Talab: natijalar/emnist_letters_cnn.keras fayli shu papkada bo'lishi kerak.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CANVAS = 280          # chizish maydoni o'lchami (piksel)
BRUSH = 11            # qalam qalinligi (radius)
CLASS_NAMES = [chr(ord("A") + i) for i in range(26)]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natijalar", "emnist_letters_cnn.keras")


def preprocess(img: Image.Image) -> np.ndarray:
    """Chizilgan rasmni EMNIST formatiga keltiradi: 28x28, oq harf qora fonda, markazda."""
    img = img.convert("L")
    bbox = img.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    letter = img.crop(bbox)
    # kvadratga keltirish (proporsiyani saqlab) va chetlardan bo'sh joy qoldirish
    side = int(max(x1 - x0, y1 - y0) * 1.25) + 4
    square = Image.new("L", (side, side), 0)
    square.paste(letter, ((side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2))
    square = square.filter(ImageFilter.GaussianBlur(radius=side / 90))
    small = square.resize((28, 28), Image.LANCZOS)
    return np.asarray(small, dtype="float32") / 255.0


def load_model():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)


def predict(model, x28: np.ndarray):
    p = model.predict(x28.reshape(1, 28, 28, 1), verbose=0)[0]
    order = p.argsort()[::-1]
    return [(CLASS_NAMES[k], float(p[k])) for k in order[:5]]


def main():
    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Qo'lyozma harfni aniqlash — EMNIST Letters CNN")
    root.configure(bg="#1e1e1e")
    root.resizable(False, False)

    status = tk.StringVar(value="Model yuklanmoqda...")
    tk.Label(root, text="Sichqoncha bilan bitta harf chizing (A–Z, katta yoki kichik)",
             fg="#dddddd", bg="#1e1e1e", font=("Segoe UI", 11)).grid(row=0, column=0, columnspan=2, pady=(10, 4))

    canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="black", highlightthickness=2,
                       highlightbackground="#555555", cursor="pencil")
    canvas.grid(row=1, column=0, padx=12, pady=6)

    right = tk.Frame(root, bg="#1e1e1e")
    right.grid(row=1, column=1, padx=(0, 12), sticky="n")
    big = tk.Label(right, text="?", fg="#4fc3f7", bg="#1e1e1e", font=("Segoe UI", 64, "bold"), width=3)
    big.pack(pady=(0, 2))
    conf = tk.Label(right, text="", fg="#aaaaaa", bg="#1e1e1e", font=("Segoe UI", 11))
    conf.pack()

    bars = tk.Canvas(right, width=230, height=130, bg="#1e1e1e", highlightthickness=0)
    bars.pack(pady=(8, 4))
    tk.Label(right, text="Modelga beriladigan 28×28 tasvir:", fg="#888888", bg="#1e1e1e",
             font=("Segoe UI", 9)).pack(pady=(6, 2))
    preview = tk.Label(right, bg="#1e1e1e")
    preview.pack()

    buttons = tk.Frame(root, bg="#1e1e1e")
    buttons.grid(row=2, column=0, columnspan=2, pady=(4, 4))
    tk.Label(root, textvariable=status, fg="#888888", bg="#1e1e1e", font=("Segoe UI", 9)).grid(
        row=3, column=0, columnspan=2, pady=(0, 8))

    image = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(image)
    state = {"last": None, "model": None, "photo": None}

    def paint(event):
        x, y = event.x, event.y
        if state["last"]:
            lx, ly = state["last"]
            canvas.create_line(lx, ly, x, y, fill="white", width=BRUSH * 2, capstyle=tk.ROUND, smooth=True)
            draw.line([lx, ly, x, y], fill=255, width=BRUSH * 2)
        canvas.create_oval(x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH, fill="white", outline="white")
        draw.ellipse([x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH], fill=255)
        state["last"] = (x, y)

    def release(_event):
        state["last"] = None
        recognize()

    def recognize():
        if state["model"] is None:
            return
        x28 = preprocess(image)
        if x28 is None:
            return
        top = predict(state["model"], x28)
        big.config(text=top[0][0])
        conf.config(text=f"ishonch: {top[0][1]:.1%}")
        bars.delete("all")
        for i, (letter, prob) in enumerate(top):
            y = 8 + i * 24
            bars.create_text(12, y + 8, text=letter, fill="#dddddd", font=("Segoe UI", 11, "bold"))
            bars.create_rectangle(30, y, 30 + 160, y + 16, fill="#333333", outline="")
            bars.create_rectangle(30, y, 30 + int(160 * prob), y + 16,
                                  fill="#4fc3f7" if i == 0 else "#607d8b", outline="")
            bars.create_text(222, y + 8, text=f"{prob:.0%}", fill="#aaaaaa", font=("Segoe UI", 9))
        pv = Image.fromarray((x28 * 255).astype("uint8")).resize((84, 84), Image.NEAREST)
        state["photo"] = ImageTk.PhotoImage(pv)
        preview.config(image=state["photo"])

    def clear():
        canvas.delete("all")
        draw.rectangle([0, 0, CANVAS, CANVAS], fill=0)
        big.config(text="?")
        conf.config(text="")
        bars.delete("all")
        preview.config(image="")
        state["photo"] = None

    style = dict(font=("Segoe UI", 11), width=12, relief="flat", cursor="hand2")
    tk.Button(buttons, text="Aniqlash", command=recognize, bg="#0288d1", fg="white", **style).pack(side="left", padx=6)
    tk.Button(buttons, text="Tozalash", command=clear, bg="#424242", fg="white", **style).pack(side="left", padx=6)

    canvas.bind("<B1-Motion>", paint)
    canvas.bind("<Button-1>", paint)
    canvas.bind("<ButtonRelease-1>", release)
    root.bind("<Escape>", lambda e: clear())
    root.bind("<Delete>", lambda e: clear())

    def load():
        try:
            state["model"] = load_model()
            status.set("Tayyor. Harf chizing — qo'yib yuborganingizda avtomatik aniqlanadi. Tozalash: Esc")
        except Exception as e:
            status.set(f"Modelni yuklab bo'lmadi: {e}")

    root.after(100, load)
    root.mainloop()


if __name__ == "__main__":
    main()
