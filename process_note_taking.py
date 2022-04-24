import argparse

from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import os

def prepare_predictor(model_name ='TF5_cond'):
    if model_name =='TF5_cond':
        model = T5ForConditionalGeneration.from_pretrained("t5-large")
        tokenizer = T5Tokenizer.from_pretrained("t5-large")

    return model, tokenizer


def load_dataset(dataset_name):
    assert dataset_name in os.listdir('original_articles'), 'dataset is not found'
    if dataset_name in os.listdir('original_articles'):
        with open('original_articles/'+ dataset_name, 'r') as fp:
            data = json.load(fp)
            return data


def generate_notes(article,model, tokenizer,token_n_per_iter = 2000, max_length_val=360, min_length_val= 120, length_penalty_val= 2.0, return_tensors="pt", file_name ='recurrent_neural_network'):
    article_length = len(article)
    current_index = 0
    note=1
    notes = article_length// token_n_per_iter
    print('number_of_notes--------->' + str(notes))
    model_name_lis = list(str(model))
    model_name = ''
    for i in model_name_lis:
        if i == '(':
            break
        model_name = model_name + i

    while current_index <article_length:
        inputs = tokenizer.encode("summarize: " + article[current_index:current_index+token_n_per_iter], return_tensors="pt", max_length=512,truncation=True)
        outputs = model.generate(inputs,max_length=max_length_val,min_length=min_length_val,
            length_penalty=length_penalty_val,num_beams=7,early_stopping=True)
        outputs = tokenizer.decode(outputs[0])
        current_index = current_index+token_n_per_iter
        note +=1
        if not os.path.isdir('notes_from_article/' +file_name + '/' + model_name):
            os.mkdir('notes_from_article/' + file_name + '/' + model_name)

        path = 'notes_from_article/' + file_name + '/' + model_name+'/'+note+' '+ str(current_index)
        os.mkdir(path)
        with open(path, 'w') as fp:
            json.dump(article, fp)
        print('note_saved')




if __name__=='__main__':
    data = load_dataset('recurrent_neural_network_based_language_model.json')
    model, tokenizer= prepare_predictor(model_name ='TF5_cond')
    generate_notes(data, model, tokenizer, token_n_per_iter=2000, max_length_val=360, min_length_val=120,
                   length_penalty_val=2.0, return_tensors="pt", file_name='recurrent_neural_network')
    print(data)









