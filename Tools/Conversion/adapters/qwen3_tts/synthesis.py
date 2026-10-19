from unittest.mock import patch

import torch
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.generation.logits_process import (
    MinNewTokensLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .source import array


def choose_token(logits, history, *, sample, temperature, top_k, top_p, repetition=1.0,
                 minimum=0, eos=None, suppressed=()):
    scores = logits.float().clone()
    if repetition != 1.0:
        scores = RepetitionPenaltyLogitsProcessor(repetition)(history, scores)
    if minimum:
        scores = MinNewTokensLengthLogitsProcessor(0, minimum, eos, device="cpu")(history, scores)
    if suppressed:
        scores = SuppressTokensLogitsProcessor(list(suppressed), device="cpu")(history, scores)
    if sample:
        if temperature != 1.0:
            scores = TemperatureLogitsWarper(temperature)(history, scores)
        if top_k:
            scores = TopKLogitsWarper(top_k)(history, scores)
        if top_p < 1.0:
            scores = TopPLogitsWarper(top_p)(history, scores)
        return torch.multinomial(torch.softmax(scores, dim=-1), 1)
    return scores.argmax(dim=-1, keepdim=True)


class CoreAIBridge:
    def __init__(self, wrapper, components):
        self.wrapper = wrapper
        self.model = wrapper.model
        self.components = components
        self.last_sequence = None

    def run(self, kind, *values):
        result = self.components.call(kind, tuple(array(value) for value in values))
        return tuple(torch.from_numpy(value) for value in result)

    def project(self, hidden):
        return self.run("projection", hidden)[0]

    def decode(self, codes):
        return self.run("codec", codes)[0]

    def predict_frame(self, previous_hidden, first_token, settings):
        talker = self.model.talker
        first_embedding = talker.get_input_embeddings()(first_token)
        prefix = torch.cat([previous_hidden.unsqueeze(1), first_embedding], dim=1)
        history = torch.empty((1, 0), dtype=torch.long)
        tokens = [first_token]
        embeddings = [first_embedding]
        for index in range(talker.config.num_code_groups - 1):
            logits, = self.run("predictor", prefix, torch.tensor([index], dtype=torch.int32))
            token = choose_token(logits, history, sample=settings["subtalker_dosample"],
                                 temperature=settings["subtalker_temperature"], top_k=settings["subtalker_top_k"],
                                 top_p=settings["subtalker_top_p"])
            tokens.append(token)
            history = torch.cat([history, token], dim=1)
            embedded = talker.code_predictor.get_input_embeddings()[index](token)
            embeddings.append(embedded)
            prefix = torch.cat([prefix, embedded], dim=1)
        return torch.cat(tokens, dim=1), torch.cat(embeddings, dim=1).sum(dim=1, keepdim=True)

    def generate(self, inputs_embeds, attention_mask, trailing_text_hidden, tts_pad_embed, **settings):
        if inputs_embeds.shape[0] != 1 or not torch.all(attention_mask == 1):
            raise ValueError("This profile requires one unpadded input")
        maximum = int(settings["max_new_tokens"])
        if not 3 <= maximum <= 128:
            raise ValueError("Generation must have a budget between 3 and 128 codec tokens")
        prefix = inputs_embeds
        history = torch.empty((1, 0), dtype=torch.long)
        states = []
        previous_hidden = None
        for step in range(maximum):
            frame = None
            if step:
                frame, embedded = self.predict_frame(previous_hidden, history[:, -1:], settings)
                text = trailing_text_hidden[:, step - 1:step] if step - 1 < trailing_text_hidden.shape[1] else tts_pad_embed
                prefix = torch.cat([prefix, embedded + text], dim=1)
            positions = torch.arange(prefix.shape[1], dtype=torch.int32).view(1, 1, -1).expand(3, 1, -1)
            logits, previous_hidden = self.run("talker", prefix, positions)
            states.append(((previous_hidden.unsqueeze(1),), frame))
            token = choose_token(logits, history, sample=settings["do_sample"], temperature=settings["temperature"],
                                 top_k=settings["top_k"], top_p=settings["top_p"],
                                 repetition=settings["repetition_penalty"], minimum=settings["min_new_tokens"],
                                 eos=settings["eos_token_id"], suppressed=settings["suppress_tokens"])
            history = torch.cat([history, token], dim=1)
            if int(token[0, 0]) == settings["eos_token_id"]:
                break
        self.last_sequence = history
        return GenerateDecoderOnlyOutput(sequences=history, hidden_states=tuple(states))

    def install(self, stack):
        stack.enter_context(patch.object(self.model.talker.text_projection, "forward", side_effect=self.project))
        stack.enter_context(patch.object(self.model.talker, "generate", side_effect=self.generate))
        stack.enter_context(patch.object(self.model.speech_tokenizer.model.decoder, "forward", side_effect=self.decode))
