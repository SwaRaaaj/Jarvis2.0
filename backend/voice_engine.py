import pyttsx3
import threading
import queue
import time
from typing import Dict, Any

class VoiceEngine:
    """Provides speech synthesis (TTS) for JARVIS audio response feedback."""

    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self._init_tts()
        
        # Worker thread for non-blocking speech execution
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _init_tts(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 175)  # Crisp speaking rate
            self.engine.setProperty('volume', 1.0)
            
            # Select male or futuristic voice if available
            voices = self.engine.getProperty('voices')
            for v in voices:
                if "david" in v.name.lower() or "zira" in v.name.lower() or "english" in v.name.lower():
                    self.engine.setProperty('voice', v.id)
                    break
        except Exception as e:
            print(f"[VoiceEngine Init Warning]: {e}")
            self.engine = None

    def speak(self, text: str):
        """Queues text to be spoken out loud."""
        if text and text.strip():
            self.speech_queue.put(text.strip())

    def stop(self):
        """Stops current speech output."""
        try:
            if self.engine:
                self.engine.stop()
        except Exception:
            pass

    def _speech_worker(self):
        while True:
            try:
                text = self.speech_queue.get()
                if text is None:
                    break
                self.is_speaking = True
                if self.engine:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as err:
                        print(f"[TTS Error]: {err}")
                self.is_speaking = False
                self.speech_queue.task_done()
            except Exception as e:
                self.is_speaking = False
                time.sleep(0.1)
