import re


def emoji_filter(text: str):
    return re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', '', text)


def symbol_filter(text: str):
    text = emoji_filter(text)
    return re.sub(
        r'[♪♫♬♩\u2600-\u26FF\u2700-\u27BF\U0001D100-\U0001D1FF「」『』【】〈〉《》〔〕〖〗〘〙〚〛｢｣\u3008-\u3011\u3014-\u301B]', '',
        text)


def repetition_filter(text: str, max_repeats=3):
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
