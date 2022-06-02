
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import os
import time



def generate_folders(folder):
    to_iterate = os.listdir('original_articles/'+str(folder))[:-2]
    return to_iterate


def summarization(folder):
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    current_time = time.time()

    for file_name in folder:
        print('test----->'+file_name)
        fine_label = file_name[:2]

        with open('original_articles/'+ file_name[0:2] + '_edited' + '/'+str(file_name), 'r') as f:
            data = json.load(f)
        current_time = time.time()
        inputs = tokenizer.encode("summarize: " +data, return_tensors="pt", max_length=350, truncation=True)
        outputs = model.generate(inputs,max_length=200,min_length= 100,length_penalty=2.0,num_beams=7,early_stopping=True)
        outputs = tokenizer.decode(outputs[0])

        if not os.path.isdir('notes_from_article/' + fine_label):
            os.mkdir('notes_from_article/' + fine_label)
        with open('notes_from_article/' + fine_label + '/'+ 'new_test'+str(file_name[3:]), 'w') as fp:
            current_time = time.time() - current_time
            print('time_for_this_article--------->' + str(current_time))
            json.dump(outputs, fp)




if __name__ =='__main__':
    with open('original_articles/P3/P3_S03.json', 'r') as f:
        data = json.load(f)
    folder = generate_folders('P3_edited')
    summarization(folder)


