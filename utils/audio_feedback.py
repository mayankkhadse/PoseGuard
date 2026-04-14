import pyttsx3
import threading
import time

engine = pyttsx3.init()
engine.setProperty('rate', 155)
engine.setProperty('volume', 1.0)

_lock           = threading.Lock()
_last_text      = ""
_last_time      = 0
_cooldown_secs  = 5


def speak(text):
    global _last_text, _last_time
    now = time.time()
    if text == _last_text and now - _last_time < _cooldown_secs:
        return
    _last_text = text
    _last_time = now

    def _run():
        with _lock:
            engine.say(text)
            engine.runAndWait()

    threading.Thread(target=_run, daemon=True).start()
