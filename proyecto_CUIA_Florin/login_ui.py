# login_ui.py
import tkinter as tk
from tkinter import messagebox
import cv2, PIL.Image, PIL.ImageTk, threading
import reconocer_caras as rc        # tu wrapper face_recognition + user_data
import user_data

class LoginWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Museo AR – Acceso")
        self.cap  = cv2.VideoCapture(0)
        w, h      = 640, 480

        self.l_video = tk.Label(self.root)
        self.l_video.pack()

        btns = tk.Frame(self.root)
        tk.Button(btns, text="Iniciar sesión",  command=self.login).pack(side="left",padx=10)
        tk.Button(btns, text="Registrarse",     command=self.register).pack(side="left",padx=10)
        btns.pack(pady=10)

        self.user_info = None        #  (username, email)
        self._update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ---------- webcam loop ------------
    def _update_frame(self):
        ok, frame = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
            self.l_video.imgtk = img
            self.l_video.configure(image=img)
        if self.user_info is None:                 # sigue mientras no haya login
            self.root.after(30, self._update_frame)

    # ---------- acciones ---------------
    def login(self):
        ok, frame = self.cap.read()
        auth = rc.authenticate(frame)
        if auth:
            uname, info = auth
            self.user_info = (uname, info["email"])
            self._quit()
        else:
            messagebox.showerror("Error", "No se reconoce tu cara. Intenta otra vez.")

    def register(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        # comprobar que hay rostro
        if not rc.encode_faces(frame):
            messagebox.showerror("Error", "No se detectó ningún rostro")
            return
        win = tk.Toplevel(self.root)
        win.title("Registro")

        tk.Label(win, text="Nombre de usuario:").grid(row=0,column=0,sticky="e")
        tk.Label(win, text="Correo electrónico:").grid(row=1,column=0,sticky="e")
        e_user = tk.Entry(win); e_mail = tk.Entry(win)
        e_user.grid(row=0,column=1); e_mail.grid(row=1,column=1)

        def _save():
            try:
                rc.register(frame, e_user.get().strip(), e_mail.get().strip())
                self.user_info = (e_user.get().strip(), e_mail.get().strip())
                win.destroy(); self._quit()
            except ValueError as e:
                messagebox.showerror("Registro", str(e))

        tk.Button(win, text="Guardar", command=_save).grid(row=2,column=0,columnspan=2,pady=5)

    def _quit(self):
        self.cap.release()
        self.root.quit()
        self.root.destroy()

# ---------- interfaz sencilla ----------
def get_user():
    w = LoginWindow()
    w.root.mainloop()
    return w.user_info              # None ↔ usuario cerró ventana

if __name__ == "__main__":
    print(get_user())
