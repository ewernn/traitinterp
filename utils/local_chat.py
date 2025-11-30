#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import torch, sys

model_id = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

has_chat_template = tokenizer.chat_template is not None
history = []

while True:
    try:
        user = input("\n> ")
    except KeyboardInterrupt:
        print("\nbye")
        break
        
    if user.lower() in ("quit", "exit"): break
    if user.lower() in ("reset", "clear", "new"):
        history = []
        print("-- cleared --")
        continue
    
    if has_chat_template:
        history.append({"role": "user", "content": user})
        inputs = tokenizer.apply_chat_template(history, return_tensors="pt", add_generation_prompt=True).to(model.device)
    else:
        # Base model: just raw completion
        inputs = tokenizer(user, return_tensors="pt").input_ids.to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs={"input_ids": inputs, "max_new_tokens": 512, "do_sample": True, "temperature": 0.7, "streamer": streamer})
    thread.start()
    
    response = ""
    try:
        for token in streamer:
            print(token, end="", flush=True)
            response += token
    except KeyboardInterrupt:
        print("\n-- interrupted --")
        if has_chat_template:
            history.pop()
        continue
    
    print()
    if has_chat_template:
        history.append({"role": "assistant", "content": response})