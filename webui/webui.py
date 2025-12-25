# This file is written by Gemini
import argparse
import re
import atexit
import json
import os
import signal
import subprocess
import sys

import gradio as gr
import platformdirs
import time
import threading

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stream_translator_gpt import __version__


class I18n:

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.locale_data = {}
        self.fallback_data = {}
        self.load_locale()

    def load_locale(self):
        locales_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")

        # Load fallback (English) first
        en_path = os.path.join(locales_dir, "en.json")
        if os.path.exists(en_path):
            try:
                with open(en_path, "r", encoding="utf-8") as f:
                    self.fallback_data = json.load(f)
            except Exception as e:
                print(f"Error loading fallback locale en: {e}")

        # Load target language if not English
        if self.lang_code != "en":
            locale_path = os.path.join(locales_dir, f"{self.lang_code}.json")
            if os.path.exists(locale_path):
                try:
                    with open(locale_path, "r", encoding="utf-8") as f:
                        self.locale_data = json.load(f)
                except Exception as e:
                    print(f"Error loading locale {self.lang_code}: {e}")
            else:
                print(f"Locale file not found: {locale_path}")
        else:
            self.locale_data = self.fallback_data

    def get(self, key):
        # Try current language first
        if key in self.locale_data:
            return self.locale_data[key]
        # Fallback to English
        if key in self.fallback_data:
            return self.fallback_data[key]
        # Not found in any locale
        print(f"Missing i18n key: {key}")
        return ""


# Global state for process management
process = None
is_running = False

# Bundled default.json location (read-only, shipped with package)
BUNDLED_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_DEFAULT_PATH = os.path.join(BUNDLED_DIR, "default.json")

# User configuration directory
USER_CONFIG_DIR = platformdirs.user_config_dir("stream-translator-gpt", appauthor=False)
USER_PRESETS_DIR = os.path.join(USER_CONFIG_DIR, "presets")
SETTINGS_FILE = os.path.join(USER_CONFIG_DIR, "settings.json")
os.makedirs(USER_PRESETS_DIR, exist_ok=True)

INPUT_KEYS = [
    "input_type", "input_url", "device_index", "device_rec_interval", "input_file", "input_format", "input_cookies",
    "input_proxy", "openai_key", "google_key", "gpt_base_url", "gemini_base_url", "overall_proxy", "model_size",
    "language", "whisper_backend", "openai_transcription_model", "vad_threshold", "min_audio_len", "max_audio_len",
    "target_audio_len", "silence_threshold", "disable_dynamic_vad", "disable_dynamic_silence", "prefix_retention_len",
    "filter_emoji", "filter_repetition", "filter_japanese_stream", "disable_transcription_context",
    "transcription_initial_prompt", "translation_prompt", "translation_provider", "gpt_model", "gemini_model",
    "history_size", "translation_timeout", "processing_proxy", "use_json_result", "retry_if_translation_fails",
    "show_timestamps", "hide_transcription", "output_file", "output_proxy", "cqhttp_url", "cqhttp_token",
    "discord_hook", "telegram_token", "telegram_chat_id", "processing_proxy_trans"
]


def get_preset_list():
    """Get list of all presets: bundled 'default' + user presets."""
    presets = ["default"]  # Always include bundled default
    if os.path.exists(USER_PRESETS_DIR):
        user_presets = [os.path.splitext(f)[0] for f in os.listdir(USER_PRESETS_DIR) if f.endswith(".json")]
        presets.extend(sorted(user_presets))
    return presets


def load_preset_data(preset_name):
    """Load preset data. 'default' loads from bundled file, others from user directory."""
    if not preset_name:
        return None
    try:
        if preset_name == "default":
            print(f"Loading bundled preset from: {BUNDLED_DEFAULT_PATH}")
            with open(BUNDLED_DEFAULT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            filename = preset_name if preset_name.endswith(".json") else preset_name + ".json"
            path = os.path.join(USER_PRESETS_DIR, filename)
            print(f"Loading user preset from: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading preset: {e}")
        return None


# Load default values from bundled default.json
DEFAULT_VALUES = load_preset_data("default") or {}


def load_settings():
    """Load global settings from settings.json."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
    return {}


def save_settings(data):
    """Save global settings to settings.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving settings: {e}")


SYSTEM_SETTINGS = load_settings()


def get_default(key, fallback=None):
    val = DEFAULT_VALUES.get(key, fallback)
    # Override with system setting for global keys
    if key == "ui_language" and "ui_language" in SYSTEM_SETTINGS:
        val = SYSTEM_SETTINGS["ui_language"]
    return val


i18n = I18n(get_default("ui_language", "en"))


def save_preset_data(preset_name, data):
    """Save preset to user presets directory. Cannot overwrite bundled 'default'."""
    if not preset_name:
        return False
    if preset_name == "default" or preset_name == "default.json":
        return False  # Cannot overwrite bundled default
    filename = preset_name if preset_name.endswith(".json") else preset_name + ".json"
    path = os.path.join(USER_PRESETS_DIR, filename)
    try:
        print(f"Saving preset to: {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception:
        return False


def delete_preset_data(preset_name):
    """Delete a user preset file. Returns True if successful, False otherwise."""
    if not preset_name:
        return False
    if preset_name == "default" or preset_name == "default.json":
        return False  # Cannot delete bundled default
    filename = preset_name if preset_name.endswith(".json") else preset_name + ".json"
    try:
        path = os.path.join(USER_PRESETS_DIR, filename)
        if os.path.exists(path):
            print(f"Deleting preset: {path}")
            os.remove(path)
            return True
        return False
    except Exception:
        return False


def cleanup():
    global process, is_running
    if process and process.poll() is None:
        print("Terminating subprocess...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    is_running = False


atexit.register(cleanup)


def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def build_translator_command(
        *,  # Enforce keyword-only arguments
        input_type,
        url,
        device_index,
        device_rec_interval,
        file_path,
        input_format,
        input_cookies,
        input_proxy,
        openai_key,
        google_key,
        overall_proxy,
        model_size,
        language,
        whisper_backend,
        openai_transcription_model,
        vad_threshold,
        min_audio_len,
        max_audio_len,
        target_audio_len,
        silence_threshold,
        disable_dynamic_vad,
        disable_dynamic_silence,
        prefix_retention_len,
        filter_emoji,
        filter_repetition,
        filter_japanese_stream,
        disable_transcription_context,
        transcription_initial_prompt,
        translation_prompt,
        translation_provider,
        gpt_model,
        gemini_model,
        history_size,
        translation_timeout,
        gpt_base_url,
        gemini_base_url,
        processing_proxy,
        use_json_result,
        retry_if_translation_fails,
        show_timestamps,
        hide_transcription,
        output_file,
        output_proxy,
        cqhttp_url,
        cqhttp_token,
        discord_hook,
        telegram_token,
        telegram_chat_id):
    cmd = [sys.executable, "-u", "-m", "stream_translator_gpt"]

    def add_arg(flag, value, default_key=None):
        """Helper to append arg if value differs from default (loaded from default.json)."""
        if value is None:
            return

        str_val = str(value)
        if default_key:
            default_val = get_default(default_key)
            # Handle special case for language 'auto' which CLI treats same as omitted
            if default_key == "language" and str_val == "auto":
                return

            # Handle list to string conversion for defaults (e.g. whisper_filters)
            if isinstance(default_val, list):
                default_str = ",".join(default_val)
                if str_val == default_str:
                    return

            if str(default_val) == str_val:
                return

            # Handle numeric mismatch (e.g. "30" vs "30.0")
            try:
                if float(str(default_val)) == float(str_val):
                    return
            except (ValueError, TypeError):
                pass

        cmd.extend([flag, str_val])

    # --- Overall (Keys & Proxy) ---
    if openai_key:
        cmd.extend(["--openai_api_key", openai_key])
    if google_key:
        cmd.extend(["--google_api_key", google_key])
    if gpt_base_url and (whisper_backend == "OpenAI Transcription API" or translation_provider == "GPT"):
        cmd.extend(["--gpt_base_url", gpt_base_url])
    if gemini_base_url and translation_provider == "Gemini":
        cmd.extend(["--gemini_base_url", gemini_base_url])
    if overall_proxy:
        cmd.extend(["--proxy", overall_proxy])

    # --- Input ---
    target_url = ""
    if input_type == "URL":
        if not url:
            return None, "Error: URL is required.\n"
        target_url = url
    elif input_type == "Device":
        target_url = "device"
        if device_index is not None:
            cmd.extend(["--device_index", str(int(device_index))])
        add_arg("--device_recording_interval", device_rec_interval, "device_rec_interval")
    elif input_type == "File":
        if not file_path:
            return None, "Error: File path is required.\n"
        target_url = file_path

    cmd.append(target_url)

    if input_type == "URL":
        if input_format:
            add_arg("--format", input_format, "input_format")
        if input_cookies:
            cmd.extend(["--cookies", input_cookies])
        if input_proxy:
            cmd.extend(["--input_proxy", input_proxy])

    # --- Audio Slicing ---
    add_arg("--vad_threshold", vad_threshold, "vad_threshold")
    if disable_dynamic_vad:
        cmd.append("--disable_dynamic_vad_threshold")

    add_arg("--min_audio_length", min_audio_len, "min_audio_len")
    add_arg("--max_audio_length", max_audio_len, "max_audio_len")
    add_arg("--target_audio_length", target_audio_len, "target_audio_len")
    add_arg("--continuous_no_speech_threshold", silence_threshold, "silence_threshold")
    add_arg("--prefix_retention_length", prefix_retention_len, "prefix_retention_len")
    if disable_dynamic_silence:
        cmd.append("--disable_dynamic_no_speech_threshold")

    # --- Transcription ---
    if whisper_backend == "Faster-Whisper":
        cmd.append("--use_faster_whisper")
    elif whisper_backend == "Simul-Streaming":
        cmd.append("--use_simul_streaming")
    elif whisper_backend == "Faster-Whisper & Simul-Streaming":
        cmd.append("--use_faster_whisper")
        cmd.append("--use_simul_streaming")
    elif whisper_backend == "OpenAI Transcription API":
        cmd.append("--use_openai_transcription_api")
        if openai_transcription_model:
            if openai_transcription_model != "gpt-4o-mini-transcribe":
                cmd.extend(["--openai_transcription_model", openai_transcription_model])

    add_arg("--model", model_size, "model_size")
    add_arg("--language", language, "language")
    if disable_transcription_context:
        cmd.append("--disable_transcription_context")

    whisper_filters = []
    if filter_emoji:
        whisper_filters.append("emoji_filter")
    if filter_repetition:
        whisper_filters.append("repetition_filter")
    if filter_japanese_stream:
        whisper_filters.append("japanese_stream_filter")

    if whisper_filters:
        add_arg("--whisper_filters", ",".join(whisper_filters), "whisper_filters")

    add_arg("--transcription_initial_prompt", transcription_initial_prompt, "transcription_initial_prompt")

    # --- Translation ---
    if translation_provider != "None":
        cmd.extend(["--translation_prompt", translation_prompt])

        if translation_provider == "GPT":
            add_arg("--gpt_model", gpt_model, "gpt_model")
        elif translation_provider == "Gemini":
            add_arg("--gemini_model", gemini_model, "gemini_model")

        add_arg("--translation_history_size", int(history_size), "history_size")
        add_arg("--translation_timeout", int(translation_timeout), "translation_timeout")

        if use_json_result:
            cmd.append("--use_json_result")
        if retry_if_translation_fails:
            cmd.append("--retry_if_translation_fails")
        if processing_proxy:
            cmd.extend(["--processing_proxy", processing_proxy])

    # --- Output ---
    if show_timestamps:
        cmd.append("--output_timestamps")
    if hide_transcription:
        cmd.append("--hide_transcribe_result")
    if output_file:
        cmd.extend(["--output_file_path", output_file])
    if output_proxy:
        cmd.extend(["--output_proxy", output_proxy])
    if discord_hook:
        cmd.extend(["--discord_webhook_url", discord_hook])
    if telegram_token and telegram_chat_id:
        cmd.extend(["--telegram_token", telegram_token])
        cmd.extend(["--telegram_chat_id", str(telegram_chat_id)])
    if cqhttp_url:
        cmd.extend(["--cqhttp_url", cqhttp_url])
        if cqhttp_token:
            cmd.extend(["--cqhttp_token", cqhttp_token])

    return cmd, None


def get_subprocess_env():
    """
    Prepare environment with project root in PYTHONPATH to ensure the child process
    can attempt to find the 'stream_translator_gpt' package even if not installed.
    """
    env = os.environ.copy()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Check if we are likely in the source tree (parent has the package)
    if os.path.isdir(os.path.join(project_root, "stream_translator_gpt")):
        # Append project_root to PYTHONPATH
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_translator(
        # Input
        input_type,
        url,
        device_index,
        device_rec_interval,
        file_path,
        input_format,
        input_cookies,
        input_proxy,
        # Keys & Overall
        openai_key,
        google_key,
        overall_proxy,
        # Transcription
        model_size,
        language,
        whisper_backend,
        openai_transcription_model,
        vad_threshold,
        min_audio_len,
        max_audio_len,
        target_audio_len,
        silence_threshold,
        disable_dynamic_vad,
        disable_dynamic_silence,
        prefix_retention_len,
        filter_emoji,
        filter_repetition,
        filter_japanese_stream,
        disable_transcription_context,
        transcription_initial_prompt,
        # Translation
        translation_prompt,
        translation_provider,
        gpt_model,
        gemini_model,
        history_size,
        translation_timeout,
        gpt_base_url,
        gemini_base_url,
        processing_proxy,
        use_json_result,
        retry_if_translation_fails,
        # Output
        show_timestamps,
        hide_transcription,
        output_file,
        output_proxy,
        cqhttp_url,
        cqhttp_token,
        discord_hook,
        telegram_token,
        telegram_chat_id):
    global process, is_running

    if is_running:
        yield "Process is already running. Please stop it first.\n"
        return

    # --- Validation ---
    if translation_provider == "GPT" and not openai_key:
        yield "Error: OpenAI API Key is required for GPT Translation.\nPlease enter your key in the 'Overall' tab.\n"
        return
    if translation_provider == "Gemini" and not google_key:
        yield "Error: Google API Key is required for Gemini Translation.\nPlease enter your key in the 'Overall' tab.\n"
        return
    if whisper_backend == "OpenAI Transcription API" and not openai_key:
        yield "Error: OpenAI API Key is required for OpenAI Transcription.\nPlease enter your key in the 'Overall' tab.\n"
        return

    # --- Logic Enforcements ---
    # CLI prioritizes Gemini if google_key is present. If user explicitly selected GPT,
    # we must ensure google_key is NOT passed to avoid accidental switch.
    if translation_provider == "GPT":
        google_key = None

    # Construct command
    cmd, error = build_translator_command(input_type=input_type,
                                          url=url,
                                          device_index=device_index,
                                          device_rec_interval=device_rec_interval,
                                          file_path=file_path,
                                          input_format=input_format,
                                          input_cookies=input_cookies,
                                          input_proxy=input_proxy,
                                          openai_key=openai_key,
                                          google_key=google_key,
                                          overall_proxy=overall_proxy,
                                          model_size=model_size,
                                          language=language,
                                          whisper_backend=whisper_backend,
                                          openai_transcription_model=openai_transcription_model,
                                          vad_threshold=vad_threshold,
                                          min_audio_len=min_audio_len,
                                          max_audio_len=max_audio_len,
                                          target_audio_len=target_audio_len,
                                          silence_threshold=silence_threshold,
                                          disable_dynamic_vad=disable_dynamic_vad,
                                          disable_dynamic_silence=disable_dynamic_silence,
                                          prefix_retention_len=prefix_retention_len,
                                          filter_emoji=filter_emoji,
                                          filter_repetition=filter_repetition,
                                          filter_japanese_stream=filter_japanese_stream,
                                          disable_transcription_context=disable_transcription_context,
                                          transcription_initial_prompt=transcription_initial_prompt,
                                          translation_prompt=translation_prompt,
                                          translation_provider=translation_provider,
                                          gpt_model=gpt_model,
                                          gemini_model=gemini_model,
                                          history_size=history_size,
                                          translation_timeout=translation_timeout,
                                          gpt_base_url=gpt_base_url,
                                          gemini_base_url=gemini_base_url,
                                          processing_proxy=processing_proxy,
                                          use_json_result=use_json_result,
                                          retry_if_translation_fails=retry_if_translation_fails,
                                          show_timestamps=show_timestamps,
                                          hide_transcription=hide_transcription,
                                          output_file=output_file,
                                          output_proxy=output_proxy,
                                          cqhttp_url=cqhttp_url,
                                          cqhttp_token=cqhttp_token,
                                          discord_hook=discord_hook,
                                          telegram_token=telegram_token,
                                          telegram_chat_id=telegram_chat_id)

    if error:
        yield error
        return
    # Start Process
    is_running = True
    start_msg = f"Running command: {subprocess.list2cmdline(cmd)}\n\n"
    log_history = [start_msg]
    yield start_msg

    try:
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1,
                                   env=get_subprocess_env(),
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)

        # Read output in a non-blocking way for the generator
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # Strip ANSI escape codes
                line = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line)
                log_history.append(line)
                yield "".join(log_history)

        rc = process.poll()
        log_history.append(f"\nProcess exited with return code {rc}\n")
        yield "".join(log_history)

    except Exception as e:
        yield f"\nException occurred: {str(e)}\n"
    finally:
        is_running = False
        process = None


def stop_translator():
    global process, is_running
    if process and is_running:
        process.terminate()
        # On Windows terminate might not look nice for console apps, but this is a python script
        return "Sending termination signal..."
    return "No running process to stop."


def run_list_devices():
    cmd = [sys.executable, "-m", "stream_translator_gpt", "device", "--list_devices"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=get_subprocess_env())
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error listing devices:\n{e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"


def run_list_formats(url, cookies, input_proxy):
    if not url:
        return "Error: URL is required to list formats."

    cmd = [sys.executable, "-m", "stream_translator_gpt", url, "--list_format"]
    if cookies:
        cmd.extend(["--cookies", cookies])
    if input_proxy:
        cmd.extend(["--input_proxy", input_proxy])

    try:
        # This might take a while, so UI might freeze slightly, but it's okay for a button click
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=get_subprocess_env())
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error listing formats:\n{e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"


# --- UI Setup ---

with gr.Blocks(title="Stream Translator GPT WebUI") as demo:
    gr.Markdown(
        f"<h1>Stream Translator GPT WebUI <small style='font-weight: normal; color: gray;'>{__version__}</small></h1>")

    with gr.Tabs():
        with gr.Tab(i18n.get("overall")):
            with gr.Group():
                with gr.Row():
                    openai_key = gr.Textbox(label=i18n.get("openai_api_key"),
                                            type="text",
                                            placeholder=i18n.get("openai_api_key_ph"))
                    google_key = gr.Textbox(label=i18n.get("google_api_key"),
                                            type="text",
                                            placeholder=i18n.get("google_api_key_ph"))
                show_api_keys = gr.Checkbox(label=i18n.get("show_api_keys"), value=True)
                with gr.Row():
                    gpt_base_url = gr.Textbox(label=i18n.get("gpt_base_url"), placeholder=i18n.get("gpt_base_url_ph"))
                    gemini_base_url = gr.Textbox(label=i18n.get("gemini_base_url"),
                                                 placeholder=i18n.get("gemini_base_url_ph"))

            with gr.Group():
                overall_proxy = gr.Textbox(label=i18n.get("overall_proxy"), placeholder=i18n.get("overall_proxy_ph"))

        with gr.Tab(i18n.get("input")):
            input_type = gr.Radio(choices=[(i18n.get("url_option"), "URL"), (i18n.get("device_option"), "Device"),
                                           (i18n.get("file_option"), "File")],
                                  label=i18n.get("input_source"),
                                  value=get_default("input_type"))

            with gr.Group(visible=True) as url_group:
                input_url = gr.Textbox(label=i18n.get("stream_url"), placeholder=i18n.get("stream_url_ph"))
                with gr.Row():
                    input_format = gr.Textbox(label=i18n.get("stream_format"),
                                              value=get_default("input_format"),
                                              placeholder=i18n.get("stream_format_ph"),
                                              scale=3)
                    list_format_btn = gr.Button(i18n.get("list_available_formats"), scale=1)
                input_cookies = gr.File(label=i18n.get("cookies_file"), type="filepath", file_count="single")
                input_proxy = gr.Textbox(label=i18n.get("input_proxy"), placeholder=i18n.get("input_proxy_ph"))

            with gr.Group(visible=False) as device_group:
                with gr.Row():
                    device_index = gr.Number(label=i18n.get("device_index"), precision=0, scale=3)
                    list_devices_btn = gr.Button(i18n.get("list_available_devices"), scale=1)
                device_rec_interval = gr.Slider(0.1,
                                                5.0,
                                                value=get_default("device_rec_interval"),
                                                label=i18n.get("recording_interval"))

            with gr.Group(visible=False) as file_group:
                input_file = gr.File(label=i18n.get("local_file_path"), type="filepath", file_count="single")

        with gr.Tab(i18n.get("audio_slicing")):
            with gr.Group():
                vad_threshold = gr.Slider(0.0, 1.0, value=get_default("vad_threshold"), label=i18n.get("vad_threshold"))
                disable_dynamic_vad = gr.Checkbox(label=i18n.get("disable_dynamic_vad"),
                                                  value=get_default("disable_dynamic_vad"))

            with gr.Group():
                with gr.Row():
                    min_audio_len = gr.Slider(0.1,
                                              10.0,
                                              value=get_default("min_audio_len"),
                                              label=i18n.get("min_audio_length"))
                    max_audio_len = gr.Slider(5.0,
                                              60.0,
                                              value=get_default("max_audio_len"),
                                              label=i18n.get("max_audio_length"))
                target_audio_len = gr.Slider(1.0,
                                             30.0,
                                             value=get_default("target_audio_len"),
                                             label=i18n.get("target_audio_length"))

                with gr.Row():
                    silence_threshold = gr.Slider(0.0,
                                                  3.0,
                                                  value=get_default("silence_threshold"),
                                                  label=i18n.get("continuous_silence_threshold"))
                    prefix_retention_len = gr.Slider(0.0,
                                                     3.0,
                                                     value=get_default("prefix_retention_len"),
                                                     label=i18n.get("prefix_retention_length"))
                disable_dynamic_silence = gr.Checkbox(label=i18n.get("disable_dynamic_silence"),
                                                      value=get_default("disable_dynamic_silence"))

        with gr.Tab(i18n.get("transcription")):
            whisper_backend = gr.Radio(choices=[
                ("Whisper", "Whisper"), ("Faster-Whisper", "Faster-Whisper"), ("Simul-Streaming", "Simul-Streaming"),
                ("Faster-Whisper & Simul-Streaming", "Faster-Whisper & Simul-Streaming"),
                (i18n.get("openai_transcription_api_option"), "OpenAI Transcription API")
            ],
                                       label=i18n.get("transcription_type"),
                                       value=get_default("whisper_backend"))

            with gr.Row():
                model_size = gr.Dropdown([
                    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large",
                    "large-v1", "large-v2", "large-v3", "large-v3-turbo"
                ],
                                         label=i18n.get("model_size"),
                                         value=get_default("model_size"),
                                         allow_custom_value=True)
                openai_transcription_model = gr.Dropdown(["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
                                                         label=i18n.get("openai_transcription_model"),
                                                         value=get_default("openai_transcription_model"),
                                                         visible=False,
                                                         allow_custom_value=True)
                language = gr.Dropdown(
                    [
                        "auto", "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs",
                        "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
                        "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko",
                        "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne",
                        "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn",
                        "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur",
                        "uz", "vi", "yi", "yo", "zh"
                    ],
                    label=i18n.get("language"),
                    value=get_default("language"),
                    allow_custom_value=True,
                    info="[Available Languages](https://github.com/openai/whisper#available-models-and-languages)")
            transcription_initial_prompt = gr.Textbox(label=i18n.get("transcription_initial_prompt"),
                                                      value=get_default("transcription_initial_prompt"),
                                                      placeholder=i18n.get("transcription_initial_prompt_ph"))
            disable_transcription_context = gr.Checkbox(label=i18n.get("disable_transcription_context"),
                                                        value=get_default("disable_transcription_context"))

            with gr.Accordion(i18n.get("filters"), open=False):
                filter_emoji = gr.Checkbox(label="Emoji Filter", value=get_default("filter_emoji"))
                filter_repetition = gr.Checkbox(label="Repetition Filter", value=get_default("filter_repetition"))
                filter_japanese_stream = gr.Checkbox(label="Japanese Stream Filter",
                                                     value=get_default("filter_japanese_stream"))

            processing_proxy_trans = gr.Textbox(label=i18n.get("processing_proxy"),
                                                placeholder=i18n.get("processing_proxy_ph"))

        with gr.Tab(i18n.get("translation")):
            translation_provider = gr.Radio(choices=[(i18n.get("none_option"), "None"), ("GPT", "GPT"),
                                                     ("Gemini", "Gemini")],
                                            label=i18n.get("llm_provider"),
                                            value=get_default("translation_provider"))

            with gr.Group(visible=False) as common_translation_group:
                translation_prompt = gr.Textbox(label=i18n.get("translation_prompt"),
                                                value=get_default("translation_prompt"),
                                                lines=2,
                                                placeholder=i18n.get("translation_prompt_ph"))

                with gr.Group(visible=False) as gpt_group:
                    gpt_model = gr.Dropdown([
                        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini",
                        "gpt-5-nano", "gpt-5.1", "gpt-5.2"
                    ],
                                            label=i18n.get("gpt_model"),
                                            value=get_default("gpt_model"),
                                            allow_custom_value=True)

                with gr.Group(visible=False) as gemini_group:
                    gemini_model = gr.Dropdown([
                        "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                        "gemini-2.5-flash-preview-09-2025", "gemini-2.5-flash-lite-preview-09-2025", "gemini-3.0-flash"
                    ],
                                               label=i18n.get("gemini_model"),
                                               value=get_default("gemini_model"),
                                               allow_custom_value=True)

                with gr.Row():
                    history_size = gr.Slider(0,
                                             10,
                                             value=get_default("history_size"),
                                             step=1,
                                             label=i18n.get("history_size"))
                    translation_timeout = gr.Number(value=get_default("translation_timeout"), label=i18n.get("timeout"))

                use_json_result = gr.Checkbox(label=i18n.get("use_json_result"), value=get_default("use_json_result"))
                retry_if_translation_fails = gr.Checkbox(label=i18n.get("retry_on_failure"),
                                                         value=get_default("retry_if_translation_fails"))

                with gr.Group():
                    processing_proxy = gr.Textbox(label=i18n.get("processing_proxy"),
                                                  placeholder=i18n.get("processing_proxy_ph"))

        with gr.Tab(i18n.get("output")):
            with gr.Row():
                show_timestamps = gr.Checkbox(label=i18n.get("output_timestamps"), value=get_default("show_timestamps"))
                hide_transcription = gr.Checkbox(label=i18n.get("hide_transcription_result"),
                                                 value=get_default("hide_transcription"))

            with gr.Group():
                output_file = gr.Textbox(label=i18n.get("save_to_file_path"),
                                         placeholder=i18n.get("save_to_file_path_ph"))

            with gr.Group():
                discord_hook = gr.Textbox(label=i18n.get("discord_webhook_url"),
                                          placeholder=i18n.get("discord_webhook_url_ph"))

            with gr.Group():
                telegram_token = gr.Textbox(label=i18n.get("telegram_token"), placeholder=i18n.get("telegram_token_ph"))
                telegram_chat_id = gr.Textbox(label=i18n.get("telegram_chat_id"),
                                              placeholder=i18n.get("telegram_chat_id_ph"))

            with gr.Group():
                cqhttp_url = gr.Textbox(label=i18n.get("cqhttp_url"), placeholder=i18n.get("cqhttp_url_ph"))
                cqhttp_token = gr.Textbox(label=i18n.get("cqhttp_token"), placeholder=i18n.get("cqhttp_token_ph"))

            with gr.Group():
                output_proxy = gr.Textbox(label=i18n.get("output_proxy"), placeholder=i18n.get("output_proxy_ph"))

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<br>")

            start_btn = gr.Button(i18n.get("run"), variant="primary")
            stop_btn = gr.Button(i18n.get("stop"), variant="stop")

            gr.Markdown("<br>")
            gr.Markdown("<br>")

            with gr.Group():
                preset_select = gr.Dropdown(choices=get_preset_list(), label=i18n.get("preset"))
                with gr.Row():
                    load_preset_btn = gr.Button(i18n.get("load"))
                    delete_preset_btn = gr.Button(i18n.get("delete"), variant="stop")

                preset_name_input = gr.Textbox(placeholder=i18n.get("name"), show_label=False)
                save_preset_btn = gr.Button(i18n.get("save"))

            gr.Markdown("<br>")
            gr.Markdown("<br>")

            ui_language = gr.Dropdown(choices=["en", "zh", "ja"],
                                      label=i18n.get("ui_language"),
                                      value=get_default("ui_language", "en"),
                                      info=i18n.get("restart_hint"),
                                      interactive=True)

        with gr.Column(scale=4):
            output_box = gr.Textbox(label=i18n.get("output_log"),
                                    lines=40,
                                    autoscroll=False,
                                    interactive=False,
                                    elem_id="output_log")

    # --- Logic connection ---

    # Input Type Visibility
    def update_input_visibility(choice):
        return {
            url_group: gr.update(visible=(choice == "URL")),
            device_group: gr.update(visible=(choice == "Device")),
            file_group: gr.update(visible=(choice == "File")),
        }

    input_type.change(update_input_visibility, input_type, [url_group, device_group, file_group])

    # Whisper Backend Visibility
    def update_backend_visibility(choice):
        openai_visible = (choice == "OpenAI Transcription API")
        return {
            openai_transcription_model: gr.update(visible=openai_visible),
            model_size: gr.update(visible=not openai_visible)
        }

    whisper_backend.change(update_backend_visibility, whisper_backend, [openai_transcription_model, model_size])

    # Translation Visibility
    def update_translation_visibility(choice):
        return {
            common_translation_group: gr.update(visible=(choice != "None")),
            gpt_group: gr.update(visible=(choice == "GPT")),
            gemini_group: gr.update(visible=(choice == "Gemini"))
        }

    translation_provider.change(update_translation_visibility, translation_provider,
                                [common_translation_group, gpt_group, gemini_group])

    # UI Language Change
    current_ui_lang = get_default("ui_language", "en")
    confirm_msg = i18n.get("restart_confirmation")
    exit_msg = i18n.get("program_exited")

    js_lang_change = f"""
    (new_val) => {{
        const current = "{current_ui_lang}";
        if (new_val === current) return new_val;
        
        let ok = confirm("{confirm_msg}");
        if (ok) {{
            setTimeout(() => {{
                window.close();
                document.body.innerHTML = "<div style='color:white; background:black; height:100vh; display:flex; flex-direction:column; justify-content:center; align-items:center; font-family:sans-serif;'><h1>" + "{exit_msg}" + "</h1></div>";
            }}, 200);
            return new_val;
        }} else {{
            return current;
        }}
    }}
    """

    def on_language_change(lang):
        if lang == current_ui_lang:
            return gr.update(value=current_ui_lang)
        SYSTEM_SETTINGS["ui_language"] = lang
        save_settings(SYSTEM_SETTINGS)
        print(f"Language changed to {lang}. Exiting...")

        def kill():
            time.sleep(0.5)
            os._exit(0)

        threading.Thread(target=kill).start()
        return gr.update(value=lang)

    ui_language.change(on_language_change, inputs=[ui_language], outputs=[ui_language], js=js_lang_change)

    # Start Action
    start_btn.click(run_translator,
                    inputs=[
                        input_type, input_url, device_index, device_rec_interval, input_file, input_format,
                        input_cookies, input_proxy, openai_key, google_key, overall_proxy, model_size, language,
                        whisper_backend, openai_transcription_model, vad_threshold, min_audio_len, max_audio_len,
                        target_audio_len, silence_threshold, disable_dynamic_vad, disable_dynamic_silence,
                        prefix_retention_len, filter_emoji, filter_repetition, filter_japanese_stream,
                        disable_transcription_context, transcription_initial_prompt, translation_prompt,
                        translation_provider, gpt_model, gemini_model, history_size, translation_timeout, gpt_base_url,
                        gemini_base_url, processing_proxy, use_json_result, retry_if_translation_fails, show_timestamps,
                        hide_transcription, output_file, output_proxy, cqhttp_url, cqhttp_token, discord_hook,
                        telegram_token, telegram_chat_id
                    ],
                    outputs=output_box,
                    concurrency_limit=1,
                    scroll_to_output=False)

    # Stop Action
    stop_btn.click(stop_translator, outputs=output_box, scroll_to_output=False)

    # List Actions
    list_devices_btn.click(run_list_devices, outputs=output_box, scroll_to_output=False)

    list_format_btn.click(run_list_formats,
                          inputs=[input_url, input_cookies, input_proxy],
                          outputs=output_box,
                          scroll_to_output=False)

    # Preset Logic
    all_inputs = [globals()[key] for key in INPUT_KEYS]

    def on_save_preset(name, *args):
        if not name:
            return gr.update()

        data = {}
        for i, key in enumerate(INPUT_KEYS):
            data[key] = args[i]

        save_preset_data(name, data)
        return gr.update(choices=get_preset_list())

    save_preset_btn.click(on_save_preset, inputs=[preset_name_input] + all_inputs, outputs=[preset_select])

    def on_load_preset(name):
        data = load_preset_data(name)
        if not data:
            return [gr.update()] * (len(all_inputs) + 1)

        updates = []
        for key in INPUT_KEYS:
            updates.append(data.get(key, get_default(key)))

        preset_name_value = "" if name == "default" else name
        updates.append(preset_name_value)

        return updates

    load_preset_btn.click(on_load_preset, inputs=[preset_select], outputs=all_inputs + [preset_name_input])

    def on_delete_preset(name):
        if not name:
            return gr.update(choices=get_preset_list()), gr.update()
        success = delete_preset_data(name)
        if not success:
            return gr.update(choices=get_preset_list()), gr.update()
        return gr.update(choices=get_preset_list()), gr.update(value=None)

    delete_confirm_msg = i18n.get("delete_confirmation")
    js_delete_confirm = f"""
    (name) => {{
        if (!name || name === 'default') return null;
        let ok = confirm("{delete_confirm_msg}");
        return ok ? name : null;
    }}
    """

    delete_preset_btn.click(on_delete_preset,
                            inputs=[preset_select],
                            outputs=[preset_select, preset_select],
                            js=js_delete_confirm)

    # Smart Scroll
    js_smart_scroll = """
    function() {
        const el = document.getElementById("output_log").querySelector("textarea");
        if (!el) return;
        
        // Use a smaller threshold (100px approx 4-5 lines)
        const threshold = 100;
        const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        
        // console.log("Scroll check:", {scrollHeight: el.scrollHeight, scrollTop: el.scrollTop, clientHeight: el.clientHeight, distance: distanceToBottom});

        if (distanceToBottom < threshold) {
            // console.log("Auto-scrolling to bottom");
            setTimeout(() => {
                el.scrollTop = el.scrollHeight;
            }, 50);
        }
    }
    """

    output_box.change(None, None, None, js=js_smart_scroll, scroll_to_output=False)

    # API Key Visibility Toggle
    def toggle_keys(show):
        new_type = "text" if show else "password"
        return gr.update(type=new_type), gr.update(type=new_type)

    show_api_keys.change(toggle_keys, inputs=show_api_keys, outputs=[openai_key, google_key])

    # Sync Processing Proxy
    processing_proxy_trans.change(fn=None, inputs=processing_proxy_trans, outputs=processing_proxy, js="(x) => x")
    processing_proxy.change(fn=None, inputs=processing_proxy, outputs=processing_proxy_trans, js="(x) => x")

    # LocalStorage Persistence
    for i, component in enumerate(all_inputs):
        key = INPUT_KEYS[i]

        js_save = f"(x) => {{ localStorage.setItem('{key}', JSON.stringify(x)); return x; }}"
        component.change(fn=None, inputs=[component], outputs=[], js=js_save)

    js_load = f"""
    (...args) => {{
        const keys = {json.dumps(INPUT_KEYS)};
        
        return args.map((defaultVal, index) => {{
            const key = keys[index];

            const stored = localStorage.getItem(key);
            if (stored === null) return defaultVal;
            try {{
                const val = JSON.parse(stored);
                if (val === null) return defaultVal;
                return val;
            }} catch (e) {{
                return defaultVal;
            }}
        }});
    }}
    """

    demo.load(
        fn=None,
        inputs=all_inputs,  # Pass default values as inputs
        outputs=all_inputs,
        js=js_load)


def main():
    parser = argparse.ArgumentParser(description="Stream Translator GPT WebUI")
    parser.add_argument("--share", action="store_true", help="Create a public link to your interface (for Colab etc.)")
    args = parser.parse_args()

    demo.queue().launch(inbrowser=True, share=args.share)


if __name__ == "__main__":
    main()
