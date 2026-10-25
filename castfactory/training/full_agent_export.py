"""Bind exported EOS to the native chat tokenizer."""
def bind_chat_generation_eos(model, tokenizer):
    """Export chat termination, not an inherited pretraining EOS default."""
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("Chat tokenizer requires one registered EOS")
    previous = model.generation_config.eos_token_id
    model.generation_config.eos_token_id = eos
    return {"previous_eos_token_id": previous, "eos_token_id": eos,
            "eos_token": tokenizer.eos_token, "source": "effective_chat_tokenizer"}
