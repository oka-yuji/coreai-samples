MODEL_PROFILES = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": ("0b6", 1024),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": ("1b7", 2048),
}


def model_profile(contract):
    try:
        return MODEL_PROFILES[contract["model_id"]]
    except (KeyError, TypeError) as error:
        raise ValueError("Use a supported Qwen3-TTS CustomVoice model identity") from error
