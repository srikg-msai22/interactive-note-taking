from transformers import T5ForConditionalGeneration, T5Tokenizer#,TransfoXLConfig, TransfoXLModel,,XLNetConfig,XLNetModel, MT5Model, XLNetTokenizer, XLMModel
import json
model = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
import os
with open('/Users/aleksandrsimonyan/Documents/GitHub/interactive-note-taking/original_articles/P1_edited/P1_S16_edited.json', 'r') as f:
    data = json.load(f)
    print(data)
    inputs = tokenizer.encode("summarize: " + data, return_tensors="pt", max_length=350, truncation=True)
    outputs = model.generate(inputs, max_length=200, min_length=100, length_penalty=2.0, num_beams=7,
                             early_stopping=True)
    outputs = tokenizer.decode(outputs[0])
    if not os.path.isdir('notes_from_article/' +'P1'):
        os.mkdir('notes_from_article/' + 'P1')
    with open('notes_from_article/' + 'P1' + '/' + 'new_test' + 'P1_S16.json', 'w') as fp:
        json.dump(outputs, fp)
