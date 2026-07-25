import re

HALLUCINATION_KEYWORDS = [
    '字幕作成',
    'この動画の字幕',
    'by ',
    'チャンネル登録',
]

HALLUCINATION_SENTENCES = {
    'ご視聴ありがとうございました',
    'ご視聴いただきありがとうございます',
    'ご視聴頂きありがとうございました',
    '字幕視聴ありがとうございました',
    '動画をご覧頂きましてありがとうございました',
    'ご覧いただきありがとうございます',
    '最後までご視聴頂きありがとうございました',
    '最後までご視聴頂き有難うございました',
    '最後までご視聴頂き有難う御座いました',
    '最後まで見ていただきありがとうございます',
    '次の動画でお会いしましょう',
    'また次回の動画でお会いしましょう',
    '次の動画もお楽しみに',
    '次回もお楽しみに',
    'エンディング',
    '次回予告',
    'またね',
    'ありがとうございました',
    'それではまた',
    'また会いましょう',
    'おわり',
    'お疲れ様でした',
    'おやすみなさい',
    'おつかれさまです',
}


def japanese_filter(text: str):
    text = re.sub(r'【.+】', '', text)
    clean_text = text.strip(" 。、！？.!?!~\t\n\r\u3000")

    for kw in HALLUCINATION_KEYWORDS:
        if kw in text:
            print('filter', text)
            return ''

    if clean_text in HALLUCINATION_SENTENCES or len(clean_text) < 3:
        print('filter', text)
        return ''

    return text
