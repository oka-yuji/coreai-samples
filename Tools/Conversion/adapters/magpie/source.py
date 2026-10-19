from pathlib import Path
import random
import re
import subprocess
import tarfile
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf, open_dict
import torch

from .artifacts import read_contract, verify_file


def source_config(path):
    with tarfile.open(path) as archive:
        candidates = [entry for entry in archive.getmembers() if entry.name.removeprefix("./") == "model_config.yaml"]
        if not candidates:
            raise ValueError("Missing model configuration")
        return OmegaConf.create(archive.extractfile(candidates[-1]).read().decode())


def check_nemo_revision(path, revision):
    root = Path(path).resolve(strict=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if actual != revision or changed:
        raise ValueError("NeMo must be an unmodified checkout of the pinned revision")
    import nemo

    if not Path(nemo.__file__).resolve().is_relative_to(root):
        raise ValueError("The imported NeMo package does not belong to the pinned checkout")


def validate_codec_weights(model, state):
    counts = {}
    for name in ("audio_encoder", "audio_decoder", "vector_quantizer"):
        module = getattr(model, name)
        selected = {key[len(name) + 1:]: value for key, value in state.items() if key.startswith(name + ".")}
        module.load_state_dict(selected, strict=True)
        counts[name] = len(selected)
    return {"strict_inference_weights": True, "state_entries": counts}


def load_model(model_archive, codec_archive, tokenizer, contract_path, nemo_root):
    from nemo.collections.tts.models import AudioCodecModel, MagpieTTSModel

    contract = read_contract(contract_path)
    check_nemo_revision(nemo_root, contract["nemo_revision"])
    verify_file(model_archive, contract["model"])
    verify_file(codec_archive, contract["codec"])
    tokenizer = Path(tokenizer).resolve(strict=True)
    if tokenizer.name != contract["tokenizer"]["revision"]:
        raise ValueError("The byte tokenizer must use the pinned snapshot directory")
    configuration = source_config(model_archive)
    if configuration.frame_stacking_factor != contract["frame_stacking_factor"]:
        raise ValueError("Unexpected frame stacking contract")
    configuration.codecmodel_path = str(Path(codec_archive).resolve())
    for spec in configuration.text_tokenizers.values():
        if spec.get("_target_") == "AutoTokenizer":
            if spec.pretrained_model != "google/byt5-small":
                raise ValueError("An unpinned tokenizer dependency was requested")
            spec.pretrained_model = str(tokenizer)
    configuration.train_ds = None
    configuration.validation_ds = None
    original_restore = AudioCodecModel.restore_from
    original_load_state = AudioCodecModel.load_state_dict

    def checked_codec_state(codec_model, state_dict, strict=True):
        result = original_load_state(codec_model, state_dict, strict=strict)
        codec_model._coreai_weight_audit = validate_codec_weights(codec_model, state_dict)
        return result

    def restore_codec(path, **kwargs):
        if Path(path).resolve() != Path(codec_archive).resolve():
            raise ValueError("An unpinned codec dependency was requested")
        codec_config = source_config(codec_archive)
        with open_dict(codec_config):
            codec_config.discriminator = None
            codec_config.use_scl_loss = False
            codec_config.train_ds = None
            codec_config.validation_ds = None
            codec_config.log_config = None
        if kwargs.pop("return_config", False):
            return codec_config
        kwargs["override_config_path"] = codec_config
        kwargs["map_location"] = torch.device("cpu")
        kwargs["strict"] = False
        return original_restore(path, **kwargs)

    torch.set_num_threads(4)
    with patch.object(AudioCodecModel, "restore_from", side_effect=restore_codec), \
            patch.object(AudioCodecModel, "load_state_dict", checked_codec_state):
        model = MagpieTTSModel.restore_from(
            str(model_archive), override_config_path=configuration,
            map_location=torch.device("cpu"), strict=True
        )
    model.eval().requires_grad_(False)
    model.use_kv_cache_for_inference = False
    if model.num_audio_codebooks != contract["codebooks"] or model.codebook_size != contract["codebook_size"]:
        raise ValueError("Unexpected audio codebook contract")
    if model.output_sample_rate != contract["sample_rate"]:
        raise ValueError("Unexpected sample rate")
    if not getattr(model._codec_model, "_coreai_weight_audit", {}).get("strict_inference_weights"):
        raise ValueError("Codec inference weights were not checked strictly")
    return model, contract


def configure_inference(model, specification):
    model.inference_parameters.max_decoder_steps = specification["max_decoder_frames"]
    model.inference_parameters.temperature = specification["temperature"]
    model.inference_parameters.topk = specification["topk"]
    model.inference_parameters.cfg_scale = specification["cfg_scale"]
    model.inference_parameters.use_LT_kv_cache = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def text_batch(model, fixture, speaker):
    from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference

    if fixture["language"] != "ja":
        raise ValueError("This research profile accepts Japanese fixtures only")
    text = re.sub(r"\s+", "", fixture["text"])
    chunks, lengths, _ = chunk_text_for_inference(
        text=text, language="ja", tokenizer_name="japanese_phoneme",
        text_tokenizer=model.tokenizer, eos_token_id=model.eos_id
    )
    if len(chunks) != 1:
        raise ValueError("This research profile accepts one short text chunk")
    return {"text": chunks[0].unsqueeze(0), "text_lens": torch.tensor([lengths[0]], dtype=torch.long),
            "speaker_indices": speaker}


def add_source_arguments(parser):
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--nemo", type=Path, required=True)


def model_from_arguments(args):
    return load_model(args.model, args.codec, args.tokenizer, args.contract, args.nemo)
