from .common import emoji_filter, symbol_filter, repetition_filter
from .chinese import chinese_filter
from .english import english_filter
from .japanese import japanese_filter

LANGUAGE_FILTERS = {
    'en': [english_filter],
    'english': [english_filter],
    'ja': [japanese_filter],
    'japanese': [japanese_filter],
    'zh': [chinese_filter],
    'chinese': [chinese_filter],
    'zh-cn': [chinese_filter],
    'zh-tw': [chinese_filter],
}


def get_language_filters(language: str = None) -> list:
    if not language or language.lower() in ('auto', 'none'):
        all_filters = []
        for flist in LANGUAGE_FILTERS.values():
            for f in flist:
                if f not in all_filters:
                    all_filters.append(f)
        return all_filters

    lang_key = language.lower()
    return LANGUAGE_FILTERS.get(lang_key, [])
