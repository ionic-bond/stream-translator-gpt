import json
import queue
import re
import threading
import time
from abc import abstractmethod
from collections import deque
from datetime import datetime, timedelta, timezone

from .common import TranslationTask, LoopWorkerBase, ClientPool, INFO


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


class LLMTranslator(LoopWorkerBase):
    PARALLEL_MAX_NUMBER = 10

    def __init__(self, model: str, prompt: str, history_size: int, use_json_result: bool, timeout: int,
                 retry_if_translation_fails: bool, debug_mode: bool = False) -> None:
        print(f'{INFO}Using {model} API as translation engine.')
        self.model = model
        self.prompt = prompt
        self.history_size = history_size
        self.use_json_result = use_json_result
        self.timeout = timeout
        self.retry_if_translation_fails = retry_if_translation_fails
        self.debug_mode = debug_mode
        self.processing_queue = deque()
        self.recent_transcripts = deque(maxlen=history_size) if history_size else None

    def _build_messages(self, translation_task: TranslationTask):
        system_prompt = 'You are a professional translator.'
        if self.use_json_result:
            system_prompt += '\nOutput the answer in json format, key is translation.'

        if self.history_size and translation_task.context_transcripts is not None:
            system_prompt += f'\n{self.prompt}'
            system_prompt += '\nThe text under "Reference" is prior context, do NOT translate it.'
            system_prompt += ' Translate ONLY the text under "Translate".'

            user_content = ''
            if translation_task.context_transcripts:
                user_content += '**Reference:**'
                for t in translation_task.context_transcripts:
                    user_content += f'\n> {t}'
                user_content += '\n'
            user_content += f'\n**Translate:**\n{translation_task.transcript}'
        else:
            user_content = f'{self.prompt}: \n{translation_task.transcript}'

        system_prompt += '\nOutput only the translation in one line, nothing else.'
        return system_prompt, user_content

    def _validate_translation(self, translation: str) -> str:
        if not translation:
            return translation
        lines = [l.strip() for l in translation.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            return lines[-1]
        return translation

    @abstractmethod
    def translate(self, translation_task: TranslationTask):
        pass

    def _prepare_context(self, task: TranslationTask):
        if self.recent_transcripts is not None:
            task.context_transcripts = list(self.recent_transcripts)
            self.recent_transcripts.append(task.transcript)

    def _trigger(self, translation_task: TranslationTask):
        if not translation_task.start_time:
            translation_task.start_time = datetime.now(timezone.utc)
        translation_task.translation_failed = False
        thread = threading.Thread(target=self.translate, args=(translation_task,))
        thread.daemon = True
        thread.start()

    def _retrigger_failed_tasks(self):
        now = datetime.now(timezone.utc)
        for task in self.processing_queue:
            if task.translation_failed and not _is_task_timeout(task, self.timeout):
                next_retry_time = getattr(task, 'next_retry_time', None)
                if next_retry_time is not None and now < next_retry_time:
                    continue
                task.retry_count = getattr(task, 'retry_count', 0) + 1
                backoff = min(2**task.retry_count, 30)
                task.next_retry_time = now + timedelta(seconds=backoff)
                self._trigger(task)
                print(f'Translation failed, retrying (backoff {backoff}s): {task.transcript}')

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
                self._prepare_context(task)
                self.processing_queue.append(task)
                self._trigger(task)
            finished_tasks = self._get_results()
            for task in finished_tasks:
                output_queue.put(task)
            if self.retry_if_translation_fails:
                self._retrigger_failed_tasks()
            time.sleep(0.1)


class GPTTranslator(LLMTranslator):

    def __init__(self, prompt_cache_key: str = None, temperature: float = None, top_p: float = None,
                 reasoning_effort: str = None, verbosity: str = None, service_tier: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prompt_cache_key = prompt_cache_key
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.service_tier = service_tier

    def translate(self, translation_task: TranslationTask):
        client = ClientPool.get_openai_client()

        system_prompt, user_content = self._build_messages(translation_task)
        if self.debug_mode:
            print(f'{INFO}[System] {system_prompt}')
            print(f'{INFO}[User] {user_content}')
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.append({'role': 'user', 'content': user_content})

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
            }

            if self.use_json_result:
                kwargs["response_format"] = {"type": "json_object"}

            match = re.match(r'^gpt-(\d+(?:\.\d+)?)', self.model)
            if match:
                version = float(match.group(1))
                if version < 5.0 or version >= 5.1:
                    kwargs["temperature"] = 0.7
                    kwargs["top_p"] = 0.9
                if version >= 5.0:
                    kwargs["reasoning_effort"] = "none" if version >= 5.1 else "minimal"

            if self.prompt_cache_key is not None:
                kwargs["prompt_cache_key"] = self.prompt_cache_key
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.reasoning_effort
            if self.verbosity is not None:
                kwargs["verbosity"] = self.verbosity
            if self.service_tier is not None:
                kwargs["service_tier"] = self.service_tier

            completion = client.chat.completions.create(**kwargs)

            translation_task.translation = completion.choices[0].message.content
            if self.debug_mode and hasattr(completion, 'usage') and completion.usage:
                print(f'{INFO}[Usage] {completion.usage}')
            if self.use_json_result:
                translation_task.translation = _parse_json_completion(translation_task.translation)
            translation_task.translation = self._validate_translation(translation_task.translation)
        except Exception as e:
            translation_task.translation_failed = True
            print(e)
            return


class GeminiTranslator(LLMTranslator):

    def __init__(self, temperature: float = None, top_p: float = None, top_k: int = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def translate(self, translation_task: TranslationTask):
        from google.genai import types

        client = ClientPool.get_google_client()

        system_prompt, user_content = self._build_messages(translation_task)
        if self.debug_mode:
            print(f'{INFO}[System] {system_prompt}')
            print(f'{INFO}[User] {user_content}')
        messages = [{'role': 'user', 'parts': [{'text': user_content}]}]

        config = types.GenerateContentConfig(
            candidate_count=1,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type='application/json' if self.use_json_result else 'text/plain',
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE')
            ])

        if self.temperature is not None:
            config.temperature = self.temperature
        if self.top_p is not None:
            config.top_p = self.top_p
        if self.top_k is not None:
            config.top_k = self.top_k

        try:
            response = client.models.generate_content(model=self.model, contents=messages, config=config)
            translation_task.translation = response.text
            if self.debug_mode and hasattr(response, 'usage_metadata') and response.usage_metadata:
                print(f'{INFO}[Usage] {response.usage_metadata}')
            if self.use_json_result:
                translation_task.translation = _parse_json_completion(translation_task.translation)
            translation_task.translation = self._validate_translation(translation_task.translation)
        except Exception as e:
            translation_task.translation_failed = True
            print(e)
            return
