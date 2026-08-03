import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import threading
import time
import os
import sys
from PIL import Image, ImageTk
import speech_recognition as sr

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from memory_vault import MemoryVault
from pc_telemetry import PCTelemetry
from screen_vision import ScreenVision
from os_automation import OSAutomation
from voice_engine import VoiceEngine
from ollama_engine import OllamaEngine

class JarvisDesktopApp:
    """Floating Windows Desktop Assistant HUD for JARVIS AI Brain with Real-time Screen Vision & Context."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JARVIS Desktop AI Brain")
        
        screen_w = self.root.winfo_screenwidth()
        window_w, window_h = 440, 680
        x_pos = max(20, screen_w - window_w - 60)
        y_pos = 60
        
        self.root.geometry(f"{window_w}x{window_h}+{x_pos}+{y_pos}")
        self.root.configure(bg="#050814")
        
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes("-topmost", True)

        self.memory = MemoryVault()
        self.vision = ScreenVision()
        self.voice = VoiceEngine()
        self.ollama = OllamaEngine(memory_vault=self.memory, screen_vision=self.vision)

        # Voice gate. Shares the cortex's instance when available so its stats show up alongside
        # the other agents; falls back to a standalone gate under the legacy engine.
        if getattr(self.ollama, "cortex", None) is not None:
            self.ears = self.ollama.cortex.ears
        else:
            from agents.ears import EarsAgent
            self.ears = EarsAgent()

        self.is_listening = False
        self.is_continuous = True
        self.is_processing = False
        self.last_active_title = ""
        self.selected_model = tk.StringVar(value="llama-3.3-70b-versatile")
        self._drag_data = {"x": 0, "y": 0}

        self.setup_ui()
        self.start_background_telemetry()
        self.start_background_vision()
        self.start_voice_listener_thread()

        self.voice.speak("JARVIS Contextual Engine online, Boss.")

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#0d1527", height=44)
        header.pack(fill=tk.X, side=tk.TOP)
        header.bind("<Button-1>", self.on_drag_start)
        header.bind("<B1-Motion>", self.on_drag_motion)

        title_lbl = tk.Label(
            header, text="🧠 JARVIS REAL-TIME AI", 
            fg="#22d3ee", bg="#0d1527", font=("Segoe UI", 11, "bold")
        )
        title_lbl.pack(side=tk.LEFT, padx=12, pady=6)
        title_lbl.bind("<Button-1>", self.on_drag_start)
        title_lbl.bind("<B1-Motion>", self.on_drag_motion)

        self.top_var = tk.BooleanVar(value=True)
        top_cb = tk.Checkbutton(
            header, text="Always Top", variable=self.top_var,
            command=lambda: self.root.attributes("-topmost", self.top_var.get()),
            fg="#94a3b8", bg="#0d1527", selectcolor="#050814",
            activebackground="#0d1527", activeforeground="#22d3ee", font=("Segoe UI", 8)
        )
        top_cb.pack(side=tk.RIGHT, padx=6)

        model_cb = ttk.Combobox(
            header, textvariable=self.selected_model,
            values=["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
            state="readonly", width=20
        )
        model_cb.pack(side=tk.RIGHT, padx=4)

        main = tk.Frame(self.root, bg="#050814", padx=12, pady=6)
        main.pack(fill=tk.BOTH, expand=True)

        top_row = tk.Frame(main, bg="#050814")
        top_row.pack(fill=tk.X, pady=4)

        self.canvas = tk.Canvas(top_row, width=110, height=110, bg="#050814", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=6)
        self.canvas.bind("<Button-1>", lambda e: self.toggle_continuous_listening())
        self.draw_orb("listening")

        vision_box = tk.LabelFrame(
            top_row, text=" 👁️ Real-time Screen Vision ", 
            fg="#34d399", bg="#050814", font=("Segoe UI", 8, "bold"), padx=4, pady=4
        )
        vision_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)

        self.vision_img_lbl = tk.Label(vision_box, bg="#020617", text="Stream Syncing...")
        self.vision_img_lbl.pack(fill=tk.BOTH, expand=True)

        self.status_lbl = tk.Label(
            main, text="🟢 HANDS-FREE VOICE & VISION ACTIVE",
            fg="#34d399", bg="#064e3b", font=("Segoe UI", 8, "bold"),
            padx=8, pady=3
        )
        self.status_lbl.pack(pady=3)

        self.transcript_lbl = tk.Label(
            main, text="Speak commands like 'Close this tab', 'What is the time?', 'Open Instagram'...",
            fg="#94a3b8", bg="#050814", font=("Segoe UI", 8, "italic"),
            wraplength=380
        )
        self.transcript_lbl.pack(pady=2)

        tele_frame = tk.Frame(main, bg="#0e172a", padx=8, pady=4)
        tele_frame.pack(fill=tk.X, pady=4)

        self.tele_lbl = tk.Label(
            tele_frame, text="Active Tab/Window: Desktop",
            fg="#22d3ee", bg="#0e172a", font=("Consolas", 8, "bold")
        )
        self.tele_lbl.pack()

        # Action Buttons
        shortcuts_frame = tk.Frame(main, bg="#050814")
        shortcuts_frame.pack(fill=tk.X, pady=4)

        actions = [
            ("❌ Close Tab", lambda: self.send_command("Close current tab")),
            ("💬 Instagram", lambda: self.prompt_instagram()),
            ("📺 YouTube", lambda: self.prompt_youtube()),
            ("✍️ Type Text", lambda: self.prompt_type_text()),
            ("⏰ What Time Is It", lambda: self.send_command("What's the current time?"))
        ]

        for text, cmd_fn in actions:
            btn = tk.Button(
                shortcuts_frame, text=text, command=cmd_fn,
                fg="#f8fafc", bg="#1e293b", activebackground="#0284c7", activeforeground="#fff",
                font=("Segoe UI", 8, "bold"), bd=0, padx=6, pady=3, cursor="hand2"
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2)

        stop_btn = tk.Button(
            shortcuts_frame, text="🛑 STOP", command=self.stop_current_task,
            fg="#ffffff", bg="#7f1d1d", activebackground="#991b1b", activeforeground="#fff",
            font=("Segoe UI", 8, "bold"), bd=0, padx=6, pady=3, cursor="hand2"
        )
        stop_btn.pack(side=tk.LEFT, padx=2, pady=2)

        # ReAct Execution Console
        log_frame = tk.LabelFrame(
            main, text=" ReAct Agent Console ",
            fg="#94a3b8", bg="#050814", font=("Segoe UI", 8, "bold"), padx=4, pady=4
        )
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        self.log_box = tk.Text(
            log_frame, bg="#020617", fg="#f1f5f9",
            font=("Consolas", 8), wrap=tk.WORD, bd=0, highlightthickness=0
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.insert(tk.END, "JARVIS Contextual Engine Ready.\n")

        # Command Entry
        entry_frame = tk.Frame(main, bg="#050814")
        entry_frame.pack(fill=tk.X, pady=4)

        self.cmd_entry = tk.Entry(
            entry_frame, bg="#0e172a", fg="#ffffff",
            font=("Segoe UI", 9), insertbackground="#22d3ee", bd=1, relief=tk.SOLID
        )
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=3)
        self.cmd_entry.bind("<Return>", lambda e: self.on_entry_submit())

        send_btn = tk.Button(
            entry_frame, text="Send", command=self.on_entry_submit,
            fg="#ffffff", bg="#0284c7", activebackground="#0369a1",
            font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=3, cursor="hand2"
        )
        send_btn.pack(side=tk.RIGHT)

    def draw_orb(self, state="idle"):
        self.canvas.delete("all")
        cx, cy, r = 55, 55, 38
        colors = {
            "listening": ("#10b981", "#059669"),
            "thinking": ("#06b6d4", "#0284c7"),
            "speaking": ("#c084fc", "#9333ea"),
            "idle": ("#38bdf8", "#4f46e5")
        }
        c1, c2 = colors.get(state, colors["idle"])
        self.canvas.create_oval(cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6, outline=c1, width=2, dash=(3, 3))
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=c2, outline=c1, width=2)
        icon = "🎙️" if state == "listening" else "🧠" if state == "thinking" else "🔊" if state == "speaking" else "⚡"
        self.canvas.create_text(cx, cy, text=icon, font=("Segoe UI", 18), fill="#ffffff")

    def set_status(self, text, bg="#064e3b", fg="#34d399", orb_state="idle"):
        self.status_lbl.config(text=text, bg=bg, fg=fg)
        self.draw_orb(orb_state)

    def append_log(self, text):
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)

    def on_drag_start(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_drag_motion(self, event):
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

    def toggle_continuous_listening(self):
        self.is_continuous = not self.is_continuous
        if self.is_continuous:
            self.set_status("🟢 HANDS-FREE LISTENING ACTIVE", "#064e3b", "#34d399", "listening")
        else:
            self.set_status("⏸️ MICROPHONE PAUSED", "#1e293b", "#94a3b8", "idle")

    def prompt_instagram(self):
        username = simpledialog.askstring("Instagram Direct", "Enter Instagram Username (leave blank for Inbox):")
        msg = simpledialog.askstring("Instagram Message", "Enter message to send (optional):")
        if username is not None:
            cmd = f"Open Instagram and chat with {username}" + (f" saying '{msg}'" if msg else "")
            self.send_command(cmd)

    def prompt_youtube(self):
        query = simpledialog.askstring("YouTube Search", "What video would you like to search?")
        if query:
            self.send_command(f"Search YouTube for {query}")

    def prompt_type_text(self):
        text = simpledialog.askstring("Type in Active Input", "Enter text to type into your currently active window/chat:")
        if text:
            self.send_command(f"Type '{text}' into my active window")

    def stop_current_task(self):
        """Kill-switch: aborts the in-flight ReAct loop before its next step and halts any speech."""
        self.ollama.cancel()
        self.voice.stop()
        self.append_log("\n[STOP] Task cancellation requested by user.")

    def on_entry_submit(self):
        cmd = self.cmd_entry.get().strip()
        if cmd and not self.is_processing:
            self.cmd_entry.delete(0, tk.END)
            self.send_command(cmd)

    def send_command(self, command_text):
        if self.is_processing:
            return
        self.is_processing = True
        self.set_status("🧠 EXECUTING TASK...", "#1e1b4b", "#818cf8", "thinking")
        self.append_log(f"\nUser: {command_text}")

        threading.Thread(target=self._process_worker, args=(command_text,), daemon=True).start()

    def _process_worker(self, user_cmd):
        model = self.selected_model.get()
        try:
            for event in self.ollama.process_command(user_cmd, model_override=model):
                ev_type = event.get("type")
                if ev_type == "status":
                    self.root.after(0, self.append_log, f"STATUS: {event.get('message')}")
                elif ev_type == "thought":
                    self.root.after(0, self.append_log, f"THOUGHT: {event.get('text')}")
                elif ev_type == "tool_exec":
                    out = event.get("output") or {}
                    summary = out.get("status", "?")
                    if out.get("matched_name"):
                        summary += f" -> '{out['matched_name']}'"
                    if out.get("on_target") is False:
                        summary += "  [OFF-TARGET]"
                    self.root.after(0, self.append_log, f"TOOL: {event.get('tool')} {summary}")
                elif ev_type == "detail":
                    # The long-form breakdown: the plan, what was actually clicked, the evidence.
                    # Shown in the console but never spoken — see NARRATOR's two registers.
                    self.root.after(0, self.append_log, f"\n--- DETAIL ---\n{event.get('text', '')}\n")
                elif ev_type == "response":
                    resp = event.get("text", "")
                    self.root.after(0, self.append_log, f"JARVIS: {resp}")
                    
                    self.root.after(0, self.set_status, "🔊 JARVIS SPEAKING...", "#4c1d95", "#c084fc", "speaking")
                    self.voice.speak(resp)
                    
                    while self.voice.is_speaking:
                        time.sleep(0.2)
        except Exception as e:
            self.root.after(0, self.append_log, f"ERROR: {e}")

        self.is_processing = False
        if self.is_continuous:
            self.root.after(0, self.set_status, "🟢 HANDS-FREE LISTENING ACTIVE", "#064e3b", "#34d399", "listening")
        else:
            self.root.after(0, self.set_status, "⚡ READY FOR COMMAND", "#0f172a", "#38bdf8", "idle")

    def start_background_vision(self):
        """Drives the HUD thumbnail from RETINA's shared feed.

        This used to be a third independent capture loop taking its own full-screen grab twice a
        second, on top of the ones RETINA and the API server were already doing. Now it subscribes
        to the single shared feed and only redraws when the screen actually changed — a static
        screen costs nothing at all.
        """
        cortex = getattr(self.ollama, "cortex", None)

        if cortex is not None:
            retina = cortex.retina

            def on_frame(frame):
                try:
                    img = self.vision.last_captured_image
                    if img is None:
                        return
                    thumb = ImageTk.PhotoImage(img.resize((150, 90), Image.Resampling.LANCZOS))

                    def update(photo):
                        self.vision_img_lbl.config(image=photo, text="")
                        self.vision_img_lbl.image = photo

                    self.root.after(0, update, thumb)
                except Exception:
                    pass

            retina.subscribe(on_frame)
            cortex.start_ambient()
            return

        # Legacy engine: no shared feed, keep the original standalone loop.
        def vision_loop():
            while True:
                try:
                    img = self.vision.capture_screen_pil()
                    if img:
                        tk_img = ImageTk.PhotoImage(img.resize((150, 90), Image.Resampling.LANCZOS))

                        def update_ui_vision(photo):
                            self.vision_img_lbl.config(image=photo, text="")
                            self.vision_img_lbl.image = photo

                        self.root.after(0, update_ui_vision, tk_img)
                except Exception:
                    pass
                time.sleep(0.5)

        threading.Thread(target=vision_loop, daemon=True).start()

    def start_voice_listener_thread(self):
        def listener_loop():
            r = sr.Recognizer()
            # Tuned for command dictation rather than the library's dictation-friendly defaults.
            r.pause_threshold = 0.6          # commands are short; 0.8s of trailing silence felt laggy
            r.phrase_threshold = 0.2         # don't discard brief utterances like "stop"
            r.non_speaking_duration = 0.4
            r.energy_threshold = 300
            # The fixed threshold was the single worst setting here: a room that gets noisier than
            # the calibration moment goes deaf, and one that gets quieter triggers on nothing.
            r.dynamic_energy_threshold = True

            scribe = getattr(getattr(self.ollama, "cortex", None), "scribe", None)

            while True:
                if self.is_continuous and not self.is_processing and not self.voice.is_speaking:
                    try:
                        with sr.Microphone() as source:
                            # Ambient noise is stable for minutes; the original code re-measured it
                            # on every iteration, blocking the mic for 0.3s each time for nothing.
                            if self.ears.calibration_due():
                                r.adjust_for_ambient_noise(source, duration=0.3)
                                self.ears.note_calibration()
                            self.root.after(0, self.transcript_lbl.config, {"text": "🎙️ Listening..."})
                            audio = r.listen(source, timeout=4, phrase_time_limit=8)

                            # show_all keeps the recogniser's whole ranked guess list plus
                            # confidences. The old call took only the top string and discarded the
                            # rest — throwing away exactly the information needed to notice that a
                            # lower-ranked alternative matches something really on screen.
                            text = ""
                            if scribe is not None:
                                try:
                                    raw = r.recognize_google(audio, show_all=True)
                                    result = scribe.transcribe(raw)
                                    text = result.text
                                    if result.changed:
                                        self.root.after(0, self.append_log,
                                                        f"[HEARD] '{result.original}' -> '{text}'")
                                except Exception:
                                    text = ""
                            if not text:
                                text = r.recognize_google(audio)
                            if not text.strip():
                                continue

                            # EARS decides whether this was aimed at JARVIS at all. Without it every
                            # transcribed utterance — including nearby conversation — became a real
                            # desktop action.
                            decision = self.ears.gate(text)
                            if not decision.dispatch:
                                self.root.after(0, self.transcript_lbl.config,
                                                {"text": f'(ignored: {decision.reason})'})
                                continue

                            self.root.after(0, self.transcript_lbl.config, {"text": f'"{text}"'})
                            self.send_command(decision.cleaned or text)
                    except Exception:
                        pass
                time.sleep(0.5)

        threading.Thread(target=listener_loop, daemon=True).start()

    def start_background_telemetry(self):
        def telemetry_loop():
            while True:
                try:
                    metrics = PCTelemetry.get_live_metrics()
                    cpu = metrics.get("cpu_percent", 0)
                    ram = metrics.get("ram_percent", 0)
                    active_win = metrics.get("active_window", {})
                    app_title = active_win.get("title", "Desktop")
                    
                    if app_title != self.last_active_title:
                        self.last_active_title = app_title
                        self.root.after(0, self.append_log, f"[Vision Focus]: {app_title[:45]}")

                    txt = f"Active: {app_title[:30]} | CPU: {cpu}% | RAM: {ram}%"
                    self.root.after(0, self.tele_lbl.config, {"text": txt})
                except Exception:
                    pass
                time.sleep(0.8)

        threading.Thread(target=telemetry_loop, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = JarvisDesktopApp(root)
    root.deiconify()
    root.lift()
    root.focus_force()
    root.attributes("-topmost", True)
    root.mainloop()
