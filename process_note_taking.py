import argparse

from transformers import T5ForConditionalGeneration, T5Tokenizer#,TransfoXLConfig, TransfoXLModel,,XLNetConfig,XLNetModel, MT5Model, XLNetTokenizer, XLMModel
import json
import os
import time

def prepare_predictor(model_name ='TF5_cond'):
    if model_name =='TF5_cond':
        model = T5ForConditionalGeneration.from_pretrained("t5-large")
        tokenizer = T5Tokenizer.from_pretrained("t5-large")

    elif model_name == 'MT5Model':
        model = T5ForConditionalGeneration.from_pretrained("google/mt5-base")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

    elif model_name =='XLNet':
        config = XLNetConfig()
        model =XLNetModel(config)
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')


    elif model_name == 'XLNETRANSFORMER':
        config =  TransfoXLConfig()
        model = TransfoXLModel(config)
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    elif model_name == 'MT5ModelXL':
        model = T5ForConditionalGeneration.from_pretrained("google/mt5-xl")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")







    return model, tokenizer


def load_dataset(dataset_name):
    assert dataset_name in os.listdir('original_articles'), 'dataset is not found'
    if dataset_name in os.listdir('original_articles'):
        with open('original_articles/'+ dataset_name, 'r') as fp:
            data = json.load(fp)
            return data


def generate_notes(article,model, tokenizer,token_n_per_iter = 2000, max_length_val=400, min_length_val= 200, length_penalty_val= 2.0, return_tensors="pt", file_name ='recurrent_neural_network', smart_iterator = False, num_articles = 10):



    article_length = len(article)
    print('article_length----->'+ str(article_length))
    if smart_iterator == True:
        token_n_per_iter = article_length //num_articles
        print('token----->' + str(token_n_per_iter))

    current_index = 0
    note=1
    notes = article_length// token_n_per_iter+1
    print('number_of_notes--------->' + str(notes))
    model_name_lis = list(str(model))
    model_name = ''
    for i in model_name_lis:
        if i == '(':
            break
        model_name = model_name + i

    model_name = 'TF5_more_notes'


    while current_index <article_length:
        current_time =time.time()
        inputs = tokenizer.encode("summarize: " + article[current_index:current_index+token_n_per_iter], return_tensors="pt", max_length=512,truncation=True)
        outputs = model.generate(inputs,max_length=max_length_val,min_length=min_length_val,
            length_penalty=length_penalty_val,num_beams=7,early_stopping=True)
        outputs = tokenizer.decode(outputs[0])
        current_index = current_index+token_n_per_iter
        if not os.path.isdir('notes_from_article/' + file_name):
            os.mkdir('notes_from_article/' + file_name)
        if not os.path.isdir('notes_from_article/' +file_name + '/' + model_name):
            os.mkdir('notes_from_article/' + file_name + '/' + model_name)

        path = 'notes_from_article/' + file_name + '/' + model_name+'/'+str(note)+'_'+ str(current_index)
        with open('notes_from_article/' + file_name + '/' + model_name +'/' + str(note)+'_'+ str(current_index) +'json', 'w') as fp:
            current_time = time.time() - current_time
            print('time_for_this_article--------->'+str(current_time ))
            json.dump(outputs, fp)

        print('note_saved----------->' +str(note))
        note +=1





if __name__=='__main__':
    name= 'regularization'
    data = load_dataset(name+ '.json')
    model, tokenizer= prepare_predictor(model_name ='TF5_cond')
    generate_notes(data, model, tokenizer, token_n_per_iter=4000, max_length_val=360, min_length_val=120,
                   length_penalty_val=2.0, return_tensors="pt", file_name=name)









