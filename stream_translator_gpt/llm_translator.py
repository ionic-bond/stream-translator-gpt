import json
import os
import queue
import threading
import time

import re
from collections import deque
from datetime import datetime, timedelta, timezone

from .common import TranslationTask, LoopWorkerBase, ApiKeyPool, INFO


# The double quotes in the values of JSON have not been escaped, so manual escaping is necessary.
def _escape_specific_quotes(input_string):
    quote_positions = [i for i, char in enumerate(input_string) if char == '"']

    if len(quote_positions) <= 4:
        return input_string

    for i in range(3, len(quote_positions) - 1):
        position = quote_positions[i]
        input_string = input_string[:position] + '\\"' + input_string[position + 1:]
        quote_positions = [pos + 1 if pos > position else pos for pos in quote_positions]

    return input_string


def _parse_json_completion(completion):
    pattern = re.compile(r'\{.*}', re.DOTALL)
    json_match = pattern.search(completion)

    if not json_match:
        return completion

    json_str = json_match.group(0)
    json_str = _escape_specific_quotes(json_str)

    try:
        json_obj = json.loads(json_str)
        translate_text = json_obj.get('translation', None)
        if not translate_text:
            return completion
        return translate_text
    except json.JSONDecodeError:
        return completion


def _is_task_timeout(task: TranslationTask, timeout: float) -> bool:
    if timeout == 0.0:
        return False
    return datetime.now(timezone.utc) - task.start_time > timedelta(seconds=timeout)


class LLMClient():

    class LLM_TYPE:
        GPT = 'GPT'
        GEMINI = 'Gemini'

    def __init__(self,
                 llm_type: str,
                 model: str,
                 prompt: str,
                 history_size: int,
                 proxy: str,
                 use_json_result: bool,
                 gemini_base_url: str = None) -> None:
        if llm_type not in (self.LLM_TYPE.GPT, self.LLM_TYPE.GEMINI):
            raise ValueError(f'Unknow LLM type: {llm_type}')
        print(f'{INFO}Using {model} API as translation engine.')
        self.llm_type = llm_type
        self.model = model
        self.prompt = prompt
        self.history_size = history_size
        self.history_messages = []
        self.proxy = proxy
        self.use_json_result = use_json_result
        self.gemini_base_url = gemini_base_url

    def _append_history_message(self, user_content: str, assistant_content: str):
        if not user_content or not assistant_content:
            return
        self.history_messages.extend([{
            'role': 'user',
            'content': user_content
        }, {
            'role': 'assistant',
            'content': assistant_content
        }])
        while (len(self.history_messages) > self.history_size * 2):
            self.history_messages.pop(0)

    def _translate_by_gpt(self, translation_task: TranslationTask):
        # https://platform.openai.com/docs/api-reference/chat/create?lang=python
        from openai import OpenAI
        import httpx

        ApiKeyPool.use_openai_api()
        client = OpenAI(http_client=httpx.Client(proxy=self.proxy))
        system_prompt = 'You are a professional translator. Translate the text accurately and concisely. Do not output any explanation or extra text.'
        if self.use_json_result:
            system_prompt += " Output the answer in json format, key is translation."
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(self.history_messages)
        user_content = f'{self.prompt}: \n{translation_task.transcript}'
        messages.append({'role': 'user', 'content': user_content})

        try:
            if self.model.startswith('gpt-4') or self.model.startswith('gpt-3'):
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"} if self.use_json_result else None,
                    temperature=0,
                    top_p=0.9,
                    stop=None if self.use_json_result else ['\n'],
                )
            else:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"} if self.use_json_result else None,
                    reasoning_effort='minimal',
                )

            translation_task.translation = completion.choices[0].message.content
            if self.use_json_result:
                translation_task.translation = _parse_json_completion(translation_task.translation)
        except Exception as e:
            translation_task.translation_failed = True
            print(e)
            return
        if self.history_size:
            self._append_history_message(user_content, translation_task.translation)

    @staticmethod
    def _gpt_to_gemini(gpt_messages: list):
        gemini_messages = []
        for gpt_message in gpt_messages:
            gemini_message = {}
            gemini_message['role'] = gpt_message['role']
            if gemini_message['role'] == 'assistant':
                gemini_message['role'] = 'model'
            gemini_message['parts'] = [{'text': gpt_message['content']}]
            gemini_messages.append(gemini_message)
        return gemini_messages

    def _translate_by_gemini(self, translation_task: TranslationTask):
        # https://ai.google.dev/tutorials/python_quickstart
        from google import genai
        from google.genai import types

        ApiKeyPool.use_google_api()

        http_options = {}
        if self.proxy:
            http_options['client_args'] = {'proxy': self.proxy}

        if self.gemini_base_url:
            http_options['base_url'] = self.gemini_base_url

        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"), http_options=http_options)

        system_prompt = 'You are a professional translator. Translate the text accurately and concisely. Do not output any explanation or extra text.'
        if self.use_json_result:
            system_prompt += " Output the answer in json format, key is translation."

        messages = self._gpt_to_gemini(self.history_messages)
        user_content = f'{self.prompt}: \n{translation_task.transcript}'
        messages.append({'role': 'user', 'parts': [{'text': user_content}]})

        config = types.GenerateContentConfig(
            candidate_count=1,
            temperature=0.0,
            top_p=0.9,
            stop_sequences=None if self.use_json_result else ['\n'],
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(include_thoughts=False),
            response_mime_type='application/json' if self.use_json_result else 'text/plain',
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE')
            ])

        try:
            response = client.models.generate_content(model=self.model, contents=messages, config=config)
            translation_task.translation = response.text
            if self.use_json_result:
                translation_task.translation = _parse_json_completion(translation_task.translation)
        except Exception as e:
            translation_task.translation_failed = True
            print(e)
            return
        if self.history_size:
            self._append_history_message(user_content, translation_task.translation)

    def translate(self, translation_task: TranslationTask):
        if self.llm_type == self.LLM_TYPE.GPT:
            self._translate_by_gpt(translation_task)
        elif self.llm_type == self.LLM_TYPE.GEMINI:
            self._translate_by_gemini(translation_task)
        else:
            raise ValueError(f'Unknow LLM type: {self.llm_type}')


class ParallelTranslator(LoopWorkerBase):
    PARALLEL_MAX_NUMBER = 10

    def __init__(self, llm_client: LLMClient, timeout: int, retry_if_translation_fails: bool):
        self.llm_client = llm_client
        self.timeout = timeout
        self.retry_if_translation_fails = retry_if_translation_fails
        self.processing_queue = deque()

    def _trigger(self, translation_task: TranslationTask):
        if not translation_task.start_time:
            translation_task.start_time = datetime.now(timezone.utc)
        translation_task.translation_failed = False
        thread = threading.Thread(target=self.llm_client.translate, args=(translation_task,))
        thread.daemon = True
        thread.start()

    def _retrigger_failed_tasks(self):
        for task in self.processing_queue:
            if task.translation_failed and not _is_task_timeout(task, self.timeout):
                self._trigger(task)
                print(f'Translation failed: {task.transcript}')
                time.sleep(1)

    def _get_results(self):
        results = []
        while self.processing_queue and (
                self.processing_queue[0].translation or _is_task_timeout(self.processing_queue[0], self.timeout) or
            (self.processing_queue[0].translation_failed and not self.retry_if_translation_fails)):
            task = self.processing_queue.popleft()
            if not task.translation:
                if _is_task_timeout(task, self.timeout):
                    print(f'Translation timeout: {task.transcript}')
                else:
                    print(f'Translation failed: {task.transcript}')
            results.append(task)
        return results

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_queue: queue.SimpleQueue[TranslationTask]):
        while True:
            if not input_queue.empty() and len(self.processing_queue) < self.PARALLEL_MAX_NUMBER:
                task = input_queue.get()
                if task is None:
                    while len(self.processing_queue) > 0:
                        finished_tasks = self._get_results()
                        for task in finished_tasks:
                            output_queue.put(task)
                        time.sleep(0.1)
                    output_queue.put(None)
                    break
                self.processing_queue.append(task)
                self._trigger(task)
            finished_tasks = self._get_results()
            for task in finished_tasks:
                output_queue.put(task)
            if self.retry_if_translation_fails:
                self._retrigger_failed_tasks()
            time.sleep(0.1)


class SerialTranslator(LoopWorkerBase):

    def __init__(self, llm_client: LLMClient, timeout: int, retry_if_translation_fails: bool):
        self.llm_client = llm_client
        self.timeout = timeout
        self.retry_if_translation_fails = retry_if_translation_fails

    def _trigger(self, translation_task: TranslationTask):
        if not translation_task.start_time:
            translation_task.start_time = datetime.now(timezone.utc)
        translation_task.translation_failed = False
        thread = threading.Thread(target=self.llm_client.translate, args=(translation_task,))
        thread.daemon = True
        thread.start()

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_queue: queue.SimpleQueue[TranslationTask]):
        current_task = None
        while True:
            if current_task:
                if (current_task.translation or current_task.translation_failed or
                        _is_task_timeout(current_task, self.timeout)):
                    if not current_task.translation:
                        if _is_task_timeout(current_task, self.timeout):
                            print(f'Translation timeout: {current_task.transcript}')
                        else:
                            print(f'Translation failed: {current_task.transcript}')
                            if self.retry_if_translation_fails:
                                self._trigger(current_task)
                                time.sleep(1)
                                continue
                    output_queue.put(current_task)
                    current_task = None

            if current_task is None and not input_queue.empty():
                current_task = input_queue.get()
                if current_task is None:
                    output_queue.put(None)
                    break
                self._trigger(current_task)
            time.sleep(0.1)
