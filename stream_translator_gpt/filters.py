import re


def emoji_filter(text: str):
    return re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', '', text)


def japanese_stream_filter(text: str):
    for filter_pattern in [
            r'【.+】', r'ご視聴ありがとうございました', r'チャンネル登録をお願いいたします', r'ご視聴いただきありがとうございます', r'チャンネル登録してね', r'字幕視聴ありがとうございました',
            r'動画をご覧頂きましてありがとうございました', r'次の動画でお会いしましょう', r'最後までご視聴頂きありがとうございました', r'次の動画もお楽しみに', r'次回もお楽しみに',
            r'また次回の動画でお会いしましょう', r'ご覧いただきありがとうございます', r'最後までご視聴頂き有難うございました', r'最後までご視聴頂き有難う御座いました', r'チャンネル登録よろしくお願いします'
    ]:
        text = re.sub(filter_pattern, '', text)

    for filter_text in ['字幕作成', 'この動画の字幕', 'by ']:
        if filter_text in text:
            print('filter', text)
            return ''

    for filter_text in ['エンディング', '次回予告', 'またね', 'ありがとうございました', 'それではまた', 'それではまた。', 'また会いましょう']:
        if filter_text == text:
            print('filter', text)
            return ''

    if len(text) < 3:
        return ''
    return text


def repetition_filter(text: str, max_repeats=3):
    """
    Filter that reduces consecutive repetitions of any substring.
    Example: "Test test test test" -> "Test test test" (if max_repeats=3)
    """
    length = len(text)
    if length < 2:
        return text

    for sub_len in range(1, length // max_repeats + 1):
        for i in range(length - sub_len * max_repeats + 1):
            substring = text[i:i + sub_len]
            if text[i:i + sub_len * max_repeats] == substring * max_repeats:
                count = 0
                curr = i
                while curr + sub_len <= length:
                    if text[curr:curr + sub_len] == substring:
                        count += 1
                        curr += sub_len
                    else:
                        break

                if count >= max_repeats:
                    keep_count = max_repeats
                    kept_text = substring * keep_count
                    return text[:i] + kept_text + text[curr:]

    return text
