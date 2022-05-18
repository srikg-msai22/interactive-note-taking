from tika import parser
import json
def extract_clean_pdf(path_to_pdf):
    raw = parser.from_file(path_to_pdf)
    st = raw['content']
    st = st.replace('\n', '')
    return st




def save_json(article_name, st):
    with open('original_articles/'+ article_name+'.json', 'w') as f:
        json.dump(st, f)


if __name__=='__main__':
    name = 'dropout'
    st = extract_clean_pdf('/Users/aleksandrsimonyan/Desktop/' + name +'.pdf')
    save_json(name, st)
