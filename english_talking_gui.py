import asyncio
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog

import edge_tts
import requests
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# ---------------- 配置 ----------------
STT_MODEL_SIZE = r"models--Systran--faster-whisper-base"
# 注意：这里填入你在 llama-server 中加载的模型名称（或者简单的 "gemma"）
LLM_MODEL = "gemma-4"
# 修改为你的 llama-server 地址（默认通常是 8080 端口）
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
# 也可以换成 "en-US-GuyNeural" (男声)
TTS_VOICE = "en-US-JennyNeural"

INPUT_AUDIO = "test_audio.wav"
OUTPUT_AUDIO = "output_response.mp3"

SAMPLE_RATE = 16000
CHANNELS = 1
# 录音设备默认使用系统默认麦克风
AUDIO_DTYPE = "int16"

# 可选：将你自己的 llama-server 启动命令写入这里。
# 留空时，点击“启动 llama-server”会弹出输入框，你可以临时填写完整命令。
LLAMA_SERVER_COMMAND = r"""
echo 使用方法: 使用 dir 命令，启动 llama_start.cmd 请保持命令窗口运行直至程序运行结束
cd
pause
""".strip()


stt_model = None


def speech_to_text(audio_path):
    segments, _ = stt_model.transcribe(audio_path, beam_size=5)
    return "".join(segment.text for segment in segments).strip()


def ask_gemma(user_text, history):
    """将完整对话历史发送给 llama-server，保留多轮上下文。"""
    system_prompt = (
        "You are a patient professional English conversation teacher. "
        "Help the student practice English. Keep the answer concise and suitable for speaking. "
        "For every student message, respond in this exact structure:\n"
        "1) Natural English: give a natural corrected version of the student's sentence. "
        "If the sentence is already correct, say it is natural and do not overcorrect.\n"
        "2) 中文解释: briefly explain the key grammar or wording corrections in Chinese. "
        "If no correction is needed, say so in Chinese.\n"
        "3) English reply: continue the conversation naturally in English and ask at most one follow-up question.\n"
        "Use clear labels. Do not reveal internal reasoning."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    response = requests.post(
        LLAMA_SERVER_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=(10, 180),
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    return (message.get("content") or "").strip()


async def text_to_speech(text, output_file=OUTPUT_AUDIO):
    if not text or not text.strip():
        raise ValueError("模型回复为空，无法生成语音。")
    # TTS 念英文部分，不朗读中文纠错说明和标签。
    english_text = extract_english_for_tts(text)
    if not english_text.strip():
        english_text = text
    communicate = edge_tts.Communicate(english_text, TTS_VOICE)
    await communicate.save(output_file)


def extract_english_for_tts(text):
    """从格式化回复中提取适合朗读的英文，避免 TTS 朗读中文说明。"""
    lines = text.splitlines()
    english_sections = []
    current = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if line.startswith(("1)", "1.", "Natural English:", "自然英文：", "自然英文:")):
            current = "correction"
            # 同行标签后的内容
            for sep in (":", "："):
                if sep in line:
                    value = line.split(sep, 1)[1].strip()
                    if value:
                        english_sections.append(value)
                    break
            continue
        if line.startswith(("2)", "2.", "中文解释：", "中文解释:")):
            current = "chinese"
            continue
        if line.startswith(("3)", "3.", "English reply:", "英文回复：", "英文回复:")):
            current = "reply"
            for sep in (":", "："):
                if sep in line:
                    value = line.split(sep, 1)[1].strip()
                    if value:
                        english_sections.append(value)
                    break
            continue
        if current in ("correction", "reply"):
            # 不让含中文的解释进入语音
            if not any('\u4e00' <= c <= '\u9fff' for c in line):
                english_sections.append(line)
    return " ".join(english_sections).strip()


class EnglishTalkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("English Talking - 英语口语练习")
        self.root.geometry("900x720")
        self.root.minsize(680, 520)

        self.events = queue.Queue()
        self.busy = False
        self.recording = False
        self.audio_stream = None
        self.audio_chunks = []
        self.audio_lock = threading.Lock()
        self.turn_history = []
        self.llama_process = None
        self.record_button_state = "idle"

        top = tk.Frame(root, padx=12, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="英语口语练习", font=("Microsoft YaHei UI", 16, "bold")).pack(side="left")
        self.start_server_button = tk.Button(
            top, text="开启 llama-server 服务", command=self.start_llama_server
        )
        self.start_server_button.pack(side="left", padx=(20, 6))
        self.server_status_var = tk.StringVar(value="服务器：未由本界面启动")
        tk.Label(top, textvariable=self.server_status_var, anchor="w").pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="正在初始化界面...")
        tk.Label(root, textvariable=self.status_var, anchor="w", padx=12).pack(fill="x")

        self.chat = scrolledtext.ScrolledText(
            root, wrap="word", font=("Microsoft YaHei UI", 11), state="disabled"
        )
        self.chat.pack(fill="both", expand=True, padx=12, pady=8)
        self.chat.tag_config("user_label", foreground="#1769aa", font=("Microsoft YaHei UI", 10, "bold"))
        self.chat.tag_config("assistant_label", foreground="#16803c", font=("Microsoft YaHei UI", 10, "bold"))
        self.chat.tag_config("error", foreground="#c62828")
        self.chat.tag_config("translation", foreground="#777777")

        controls = tk.Frame(root, padx=12, pady=8)
        controls.pack(fill="x")
        self.record_button = tk.Button(
            controls, text="🎙 开始录音", font=("Microsoft YaHei UI", 12),
            height=2, width=18, command=self.toggle_recording, state="disabled"
        )
        self.record_button.pack(side="left")
        tk.Button(controls, text="清空对话", command=self.clear_chat).pack(side="left", padx=8)
        tk.Button(controls, text="重置对话上下文", command=self.reset_history).pack(side="left")
        tk.Label(
            controls,
            text="点击开始录音，再次点击结束；处理完成并播放后可继续下一轮。",
            anchor="w", justify="left", wraplength=390,
        ).pack(side="left", padx=12)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.poll_events)
        self.root.after(200, self.initialize_model)

    def append_chat(self, label, text, tag):
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{label}\n", tag)
        self.chat.insert("end", f"{text}\n\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def clear_chat(self):
        if self.busy or self.recording:
            messagebox.showinfo("请稍候", "录音或处理期间不能清空对话。")
            return
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")

    def reset_history(self):
        if self.busy or self.recording:
            messagebox.showinfo("请稍候", "录音或处理期间不能重置上下文。")
            return
        self.turn_history.clear()
        self.set_status("已重置模型对话上下文；屏幕上的历史文字仍保留。")

    def set_status(self, text):
        self.status_var.set(text)

    def initialize_model(self):
        self.record_button.configure(state="disabled")
        self.set_status("正在加载 Whisper 模型，请稍候...")
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        try:
            global stt_model
            stt_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")
            self.events.put(("ready", "Whisper 模型已加载，可以开始录音。"))
        except Exception as exc:
            self.events.put(("error", f"Whisper 模型加载失败：{exc}"))

    def toggle_recording(self):
        if self.busy:
            return
        if not self.recording:
            self.begin_recording()
        else:
            self.end_recording()

    def begin_recording(self):
        try:
            self.audio_chunks = []
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=AUDIO_DTYPE,
                callback=self.audio_callback,
            )
            self.audio_stream.start()
            self.recording = True
            self.record_button.configure(text="■ 结束录音")
            self.set_status("正在录音……再次点击按钮结束录音。")
        except Exception as exc:
            self.audio_stream = None
            self.recording = False
            messagebox.showerror("录音启动失败", str(exc))

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            self.events.put(("status", f"录音设备提示：{status}"))
        with self.audio_lock:
            self.audio_chunks.append(indata.copy())

    def end_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.record_button.configure(text="正在处理...", state="disabled")
        try:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
            with self.audio_lock:
                chunks = list(self.audio_chunks)
                self.audio_chunks = []
            if not chunks:
                raise RuntimeError("没有录到音频，请检查麦克风。")
            import numpy as np
            audio = np.concatenate(chunks, axis=0)
            sf.write(INPUT_AUDIO, audio, SAMPLE_RATE, subtype="PCM_16")
            self.busy = True
            self.set_status("录音已保存，正在识别语音...")
            threading.Thread(target=self._turn_worker, daemon=True).start()
        except Exception as exc:
            self.busy = False
            self.record_button.configure(text="🎙 开始录音", state="normal")
            messagebox.showerror("录音结束失败", str(exc))

    def _turn_worker(self):
        try:
            user_text = speech_to_text(INPUT_AUDIO)
            if not user_text:
                self.events.put(("error", "没有识别到语音，请重新录音。"))
                return
            self.events.put(("user", user_text))
            self.events.put(("status", "正在等待 Gemma 回复..."))
            reply = ask_gemma(user_text, self.turn_history)
            if not reply:
                raise RuntimeError("模型返回空回复，请检查 llama-server。")
            self.turn_history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ])
            self.events.put(("assistant", reply))
            self.events.put(("status", "正在生成语音..."))
            asyncio.run(text_to_speech(reply, OUTPUT_AUDIO))
            self.events.put(("status", "正在播放 AI 英文回复..."))
            self.play_audio_blocking(OUTPUT_AUDIO)
            self.events.put(("status", "本轮完成，可以继续点击“开始录音”。"))
        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self.events.put(("turn_done", None))

    @staticmethod
    def play_audio_blocking(path):
        import pygame
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(os.path.abspath(path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        finally:
            pygame.mixer.music.stop()
            pygame.mixer.quit()

    def start_llama_server(self):
        if self.llama_process and self.llama_process.poll() is None:
            messagebox.showinfo("llama-server", "本程序启动的 llama-server 仍在运行。")
            return
        command = LLAMA_SERVER_COMMAND
        if not command or all(line.strip().startswith("#") or not line.strip() for line in command.splitlines()):
            command = tk.simpledialog.askstring(
                "启动 llama-server",
                "请输入完整启动命令（例如 cd /d 路径 && llama-server.exe ...）：",
                parent=self.root,
            )
        if not command or not command.strip():
            return
        try:
            # 用 cmd.exe 执行，允许用户填入 cd /d、&& 等 Windows 命令。
            self.llama_process = subprocess.Popen(
                ["cmd.exe", "/k", command],
                cwd=os.getcwd(),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            self.server_status_var.set("服务器：已请求启动（请查看新开的控制台）")
            self.set_status("已打开 llama-server 控制台；请确认模型加载完成后再开始对话。")
        except Exception as exc:
            messagebox.showerror("启动失败", str(exc))

    def poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "ready":
                    self.set_status(payload)
                    self.record_button.configure(state="normal")
                elif kind == "status":
                    self.set_status(payload)
                elif kind == "user":
                    self.append_chat("你（语音识别）", payload, "user_label")
                elif kind == "assistant":
                    self.append_chat("AI 老师（英文纠错 + 中文解释）", payload, "assistant_label")
                elif kind == "error":
                    self.set_status(payload)
                    self.append_chat("错误", payload, "error")
                    messagebox.showerror("程序运行错误", payload)
                elif kind == "turn_done":
                    self.busy = False
                    self.record_button.configure(text="🎙 开始录音", state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def on_close(self):
        if self.recording and self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
        # 不自动终止 llama-server，以免误杀用户本来需要继续使用的服务。
        self.root.destroy()


def main():
    root = tk.Tk()
    # askstring 位于 tkinter.simpledialog
    import tkinter.simpledialog
    app = EnglishTalkingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
