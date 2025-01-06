import spacy
import re
from wordcloud import WordCloud
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import sys

# ATTENTION! This operation is extremely slow and may take hours to complete
# if executed on the entire dataset. It is best to use a subset of the data.
# The output of a full execution is available in data/cleaned_data_2023.csv.

custom_stopwords = {'excalexpre', 'jfildefxml', 'definition', 'linkbaseexpreexdef', 'jfilxsd', 
                    'linkbaseexcalexpre', 'presentation', 'linkbaseexdef', 'exdef', 'jfilprexml', 
                    'xbrl', 'schemaexlabexcal', 'jfilcalxml', 'expreexdef', 'exexsch', 'linkbase',
                    'headerfilestats', 'filename', 'kedgardata', 'txtfilename', 'grossfilesizegrossfilesize', 'netfilesizenetfilesize', 'nontextdocumenttypecharsnontextdocumenttypechars', 'htmlcharshtmlchars', 'xbrlcharsxbrlchars', 'xmlcharsxmlchar', 'nexhibitsnexhibitsfilestatssec', 'header', 'hdrsgml', 'datetimeaccession', 'kpublic', 'countconformed', 'reportfile', 'datedate',
                     'changefiler', 'datum', 'actsec', 'txt', 'jfilkhtm', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'i', 'ii', 'iii', 'iiii', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx', 'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii', 'xxviii', 'xxix', 'xxx', 'xxxi', 'exex', 'jfilexhtm', 'rd'}

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
sys.stdout.write('\rLoading Data...')
df = pd.read_csv('data/updated_data_2023.csv').set_index('File')
nlp.max_length = 1e9
for word in custom_stopwords:
    nlp.vocab[word].is_stop = True

def update_progress(index, total, message="Processing file"):
    progress = f"{message} {index}/{total}..."
    sys.stdout.write('\r' + progress)
    sys.stdout.flush()

def process_text(file_path):
    path, text = file_path # TODO: Read the file
    doc = nlp(text)
    filtered_tokens = [
    token.text.lower()
    for token in doc
    if not token.is_stop and token.is_alpha]

    return (path, filtered_tokens)

if __name__ == '__main__':
    texts = df['Text'].tolist()
    df.drop('Text', axis=1, inplace=True)
    total_files = len(texts)
    texts = [(path, text) for path, text in zip(df.index.to_list(), texts)]
    df['Tokens'] = [[] for _ in range(len(df))]

    with Pool() as pool:
        results = []
        for idx, result in enumerate(pool.imap(process_text, texts), 1):
            results.append(result)
            update_progress(idx, total_files, 'Cleaning text')

    for path, tokens in results:
        try:
            df.at[path, 'Tokens'] = tokens
        except Exception as e:
            print(e)


    sys.stdout.write('\rProcessing complete! ' + ' ' * 50 + '\n')

    df = df.reset_index().dropna()
    df.to_csv('data/subset_tokens_2023.csv', index=False)
    print(df)