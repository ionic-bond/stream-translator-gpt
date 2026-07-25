from .common import emoji_filter, symbol_filter, repetition_filter
from .japanese import japanese_filter

LANGUAGE_FILTERS = {
    'ja': [japanese_filter],
    'japanese': [japanese_filter],
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
