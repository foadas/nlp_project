import os
from nltk.stem import PorterStemmer

project_dir = os.getcwd()


def read_file(path):
    path = os.path.join(project_dir, path)
    with open(path, 'r', encoding='utf-8-sig') as file:
        file_contents = file.read()
    return file_contents


def read_lines(path):
    path = os.path.join(project_dir, path)
    with open(path, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return lines


def spell_errors(path):
    data = read_lines(path)
    spell_errors = {}
    for line in data:
        line = line.strip()
        word, errors = line.split(':')
        errors_array = [v.strip() for v in errors.split(',')]
        spell_errors[word] = errors_array
    return spell_errors


def write_file(file_name, data):
    path = os.path.join(project_dir, f'files/TextProcessing/output/{file_name}')
    data_type = type(data)
    with open(path, 'w') as file:
        if data_type == 'list':
            for item in data:
                file.write(str(item) + '\n')
        if data_type == dict:
            for key, value in data.items():
                file.write(f'{key} : {value}\n')


def tokenizer(text):
    punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*',
                   '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
                   '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    for mark in punctuation:
        text = text.replace(mark, f' {mark} ')

    tokens = text.split()
    return tokens


def count_of_words(text):
    tokens = tokenizer(text)
    token_count = {}
    for item in tokens:
        if item in token_count:
            token_count[item] += 1
        else:
            token_count[item] = 1
    write_file('count', token_count)


def stemming(text):
    tokens = tokenizer(text)
    stemmer = PorterStemmer()
    stemmed_tokens = {}
    for item in tokens:
        stemmed_tokens[item] = stemmer.stem(item)
    write_file('stemming', stemmed_tokens)


def spell_correction():
    words = read_file('files/SpellCorrection/test/spell-testset.txt').split()
    errors = spell_errors('files/SpellCorrection/spell-errors.txt')
    candidates = {}
    for word in words:
        candidates_array = []
        for key, values in errors.items():
            if key == word:
                candidates_array.append(word)
            for value in values:
                if value == word:
                    candidates_array.append(key)
                    break
            candidates[word] = candidates_array
    #print(candidates)


if __name__ == '__main__':
    print('1.preprocessing', '\n2.spell correction, \n3.text classification')
    base_menu = input()
    if base_menu == '1':
        text = read_file('files/TextProcessing/61085-0.txt')
        print('1.tokenization', '\n2.lowercase folding, \n3.count of words')
        menu1 = input()
        if menu1 == '1':
            tokens = tokenizer(text)
            write_file('tokens', tokens)
            # print(len(tokens))
        elif menu1 == '2':
            print(text.lower())
        elif menu1 == '3':
            count_of_words(text)
        elif menu1 == '4':
            stemming(text)
    if base_menu == '2':
        menu2 = input()
        if menu2 == '1':
            spell_correction()
