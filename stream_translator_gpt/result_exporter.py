import os
import queue
import requests

from .common import TranslationTask, LoopWorkerBase, sec2str


def _send_to_cqhttp(url: str, token: str, proxies: dict, text: str):
    headers = {'Authorization': 'Bearer {}'.format(token)} if token else None
    data = {'message': text}
    try:
        requests.post(url, headers=headers, data=data, timeout=10, proxies=proxies)
    except Exception as e:
        print(e)


def _send_to_discord(webhook_url: str, proxies: dict, text: str):
    for sub_text in text.split('\n'):
        data = {'content': sub_text}
        try:
            requests.post(webhook_url, json=data, timeout=10, proxies=proxies)
        except Exception as e:
            print(e)


def _send_to_telegram(token: str, chat_id: int, proxies: dict, text: str):
    url = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(token, chat_id, text)
    try:
        requests.post(url, timeout=10, proxies=proxies)
    except Exception as e:
        print(e)


def _output_to_file(file_path: str, text: str):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n\n')


class ResultExporter(LoopWorkerBase):

    def __init__(self, output_file_path: str) -> None:
        if output_file_path:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_whisper_result: bool,
             output_timestamps: bool, proxy: str, output_file_path: str, cqhttp_url: str, cqhttp_token: str,
             discord_webhook_url: str, telegram_token: str, telegram_chat_id: int):
        proxies = {"http": proxy, "https": proxy} if proxy else None
        while True:
            task = input_queue.get()
            timestamp_text = '{} --> {}'.format(sec2str(task.time_range[0]), sec2str(task.time_range[1]))
            text_to_send = (task.transcribed_text + '\n') if output_whisper_result else ''
            if output_timestamps:
                text_to_send = timestamp_text + '\n' + text_to_send
            if task.translated_text:
                text_to_print = task.translated_text
                if output_timestamps:
                    text_to_print = timestamp_text + ' ' + text_to_print
                text_to_print = text_to_print.strip()
                print('\033[1m{}\033[0m'.format(text_to_print))
                text_to_send += task.translated_text
            text_to_send = text_to_send.strip()
            if output_file_path:
                _output_to_file(output_file_path, text_to_send)
            if telegram_token and telegram_chat_id:
                _send_to_telegram(telegram_token, telegram_chat_id, proxies, text_to_send)
            if discord_webhook_url:
                _send_to_discord(discord_webhook_url, proxies, text_to_send)
            if cqhttp_url:
                _send_to_cqhttp(cqhttp_url, cqhttp_token, proxies, text_to_send)
