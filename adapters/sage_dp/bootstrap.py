"""Disable the fork's eager CLIP download for explicitly language-free policies."""
import sys
import types

def disable_unused_language_encoder():
    module=types.ModuleType('robomimic.utils.lang_utils')
    module.LANG_EMB_OBS_KEY='lang_emb'
    def get_lang_emb(text):
        if text is not None:raise RuntimeError('This bounded DP adapter does not enable language conditioning')
        return None
    def get_lang_emb_shape():
        raise RuntimeError('Language encoder is intentionally disabled for this RGB/state DP')
    module.get_lang_emb=get_lang_emb;module.get_lang_emb_shape=get_lang_emb_shape
    sys.modules[module.__name__]=module
