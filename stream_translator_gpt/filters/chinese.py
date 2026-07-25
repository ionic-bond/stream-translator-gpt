import re


# Common Whisper silence/noise hallucinations, mostly derived from video outro
# subtitles present in the training data.
HALLUCINATION_KEYWORDS = [
    '字幕由',
    '字幕制作',
    '字幕提供',
    '本字幕',
    'amara.org',
    '感谢观看',
    '谢谢观看',
    '感谢收看',
    '谢谢收看',
    '欢迎订阅',
    '请订阅',
    '订阅我的频道',
    '订阅频道',
]

HALLUCINATION_SENTENCES = {
    '感谢大家观看',
    '谢谢大家观看',
    '感谢您的观看',
    '谢谢您的观看',
    '下期再见',
    '下集再见',
    '下次再见',
    '我们下期再见',
    '我们下集再见',
    '再见',
    '拜拜',
    '结束',
}


def chinese_filter(text: str):
    text = re.sub(r'【.+?】', '', text)
    clean_text = text.strip(" 。、！？.!?!~\t\n\r\u3000")

    if any(keyword in clean_text for keyword in HALLUCINATION_KEYWORDS):
        print('filter', text)
        return ''

    if clean_text in HALLUCINATION_SENTENCES or len(clean_text) < 3:
        print('filter', text)
        return ''

    return text
