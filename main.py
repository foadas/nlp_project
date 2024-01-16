import json
import math
import os
from nltk.stem import PorterStemmer
import enchant
from collections import Counter

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
    dataset = read_file('files/SpellCorrection/test/Dictionary/Dataset.data').split(' ')
    del_matrix = read_file('files/SpellCorrection/test/Confusion Matrix/del-confusion.data').replace("'", '"')
    ins_matrix = read_file('files/SpellCorrection/test/Confusion Matrix/ins-confusion.data').replace("'", '"')
    sub_matrix = read_file('files/SpellCorrection/test/Confusion Matrix/sub-confusion.data').replace("'", '"')
    trans_matrix = read_file('files/SpellCorrection/test/Confusion Matrix/Transposition-confusion.data').replace("'",
                                                                                                                 '"')
    counter = Counter(dataset)


    candidates = {}
    probs = {}
    d = enchant.Dict("en_US")
    for word in words:
        suggestions = d.suggest(word)
        filtered_candidates = [candid.lower() for candid in suggestions if
                               not any(punctuation in candid for punctuation in [' ', '-', "'"])]
        candidates[word] = filtered_candidates
        # print(candidates[word])
    # print(candidates)

    for key, values in candidates.items():
        max_value = 0
        for value in values:
            changes = min_edit_distance(key, value)

            if changes[0] <= 1:
                if key not in probs:
                    probs[key] = ''
                # print('key', key)
                # print('value', value)
                channel = channel_model(changes[1], dataset, del_matrix, ins_matrix, sub_matrix, trans_matrix, counter)
                lang = language_model(value, dataset)
                current_value = channel * lang * (10 ** 9)
                if probs[key] == '' or current_value > max_value:
                    max_value = current_value
                    probs[key] = value

    prob_path = os.path.join(project_dir, 'files/SpellCorrection/correction.txt')
    json_data = json.dumps(probs, indent=2)
    with open(prob_path, 'w') as prob_file:
        prob_file.write(json_data)


def min_edit_distance(word1, word2):
    # turning 1 to 2
    len_word1 = len(word1)
    len_word2 = len(word2)

    dp = [[0] * (len_word2 + 1) for _ in range(len_word1 + 1)]
    # print(dp)
    operations = [[""] * (len_word2 + 1) for _ in range(len_word1 + 1)]
    # print(operations)

    for i in range(len_word1 + 1):
        dp[i][0] = i
        operations[i][0] = 'D'
    for j in range(len_word2 + 1):
        dp[0][j] = j
        operations[0][j] = 'I'

    for i in range(1, len_word1 + 1):
        for j in range(1, len_word2 + 1):
            cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + cost,  # Deletion
                dp[i][j - 1] + cost,  # Insertion
                dp[i - 1][j - 1] + (0 if word1[i - 1] == word2[j - 1] else cost)  # Substitution
            )
            if i > 1 and j > 1 and word1[i - 1] == word2[j - 2] and word1[i - 2] == word2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)

            # print(i,j,dp[i][j])
            if dp[i][j] == dp[i - 1][j] + cost:
                operations[i][j] = "D"  # Deletion
            elif dp[i][j] == dp[i][j - 1] + cost:
                operations[i][j] = "I"  # Insertion
            elif dp[i][j] == dp[i - 1][j - 1] + (0 if word1[i - 1] == word2[j - 1] else cost):
                operations[i][j] = "S" if word1[i - 1] != word2[j - 1] else ""  # Substitution

            if i > 1 and j > 1 and word1[i - 1] == word2[j - 2] and word1[i - 2] == word2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)
                if dp[i][j] == dp[i - 2][j - 2] + cost:
                    operations[i][j] = "T"
            # print(i,j,operations[i][j])
    i, j = len_word1, len_word2
    distance = dp[i][j]
    if distance > 1:
        return [2, '']

    changes = {}

    while i > 0 or j > 0:
        operation = operations[i][j]
        # print(i, j, operation)
        if operation == "D":
            if i != 1:
                changes["insert"] = f'{word1[i - 2]}|{word1[i - 1]}'
            else:
                changes["insert"] = f'{word1[i]}|{word1[i - 1]}'
            # .append(f"Delete '{word1[i - 1]}' at position {i}")
            i -= 1
        elif operation == "I":
            if j != 1:
                changes["delete"] = f'{word2[j - 2]}|{word2[j - 1]}'
            else:

                # print(j)
                changes["delete"] = f'{word2[j]}|{word2[j - 1]}'

            # changes.append(f"Insert '{word2[j - 1]}' at position {i + 1}")
            j -= 1
        elif operation == "S":
            changes["sub"] = f'{word1[i - 1]}|{word2[i - 1]}'
            # changes.append(f"Replace '{word1[i - 1]}' with '{word2[j - 1]}' at position {i}")
            i -= 1
            j -= 1
        elif operation == "T":
            changes["trans"] = f'{word1[i - 2:i]}'
            # changes.append(f"Transpose '{word1[i - 2:i]}' with '{word2[j - 2:j]}' at positions {i - 1} and {i}")
            i -= 2
            j -= 2
        else:
            i -= 1
            j -= 1

    return [distance, changes]


def channel_model(xw, dataset, del_matrix, ins_matrix, sub_matrix, trans_matrix, counter):
    if 'delete' in xw:
        w = xw['delete'].split('|')[1]
        x = xw['delete'].split('|')[0]
        matrix = json.loads(del_matrix)
        matrix_value = matrix[f'{x + w}']
        #count = 1
        # print(dataset)
        # dataset = ['technologies', 'esssss', 'fffesee', 'example', 'yes', 'guess']
        count = counter[f'{x + w}'] + 1
        #print('del', count)
        # print(matrix_value)

    elif 'insert' in xw:

        x = xw['insert'].split('|')[0]
        w = xw['insert'].split('|')[1]
        matrix = json.loads(ins_matrix)
        # print(x, w)
        matrix_value = matrix[f'{x + w}']
        # print(f'{x}')
        count = counter[f'{x}'] + 1
        #print('in', count)
        # print(count)
        # print(matrix_value)

    elif 'trans' in xw:

        w = xw['trans'][1]
        x = xw['trans'][0]
        matrix = json.loads(trans_matrix)
        matrix_value = matrix[f'{x + w}']
        count = counter[f'{x + w}'] + 1
        #print('t', count)
        # print(count)
        # print(matrix_value)

    elif 'sub' in xw:

        w = xw['sub'].split('|')[1]
        x = xw['sub'].split('|')[0]
        matrix = json.loads(sub_matrix)
        matrix_value = matrix[f'{x + w}']
        # print(f'{w}')
        count = counter[f'{w}'] + 1
        #print('sub',count)
        # print(matrix_value)
    else:
        matrix_value = 95
        count = 100
    # print(xw)
    # print(matrix_value)
    # print(count)
    # print((matrix_value / count) * 10 ** 9)
    return matrix_value / count


def language_model(w, dataset):
    count = 0
    for word in dataset:
        if w == word:
            count += 1

    return count / len(dataset)


def classification_dictionary():
    classes = ['Comp.graphics', 'rec.autos', 'sci.electronics', 'soc.religion.christian', 'talk.politics.mideast']
    words_list = set()
    class_number = {}
    class_size = {}
    class_words = {}
    count_of_words_in_class = {}
    count = 0
    count_all = 0
    for directory in classes:
        path = os.path.join(project_dir, f'files/Classification-Train And Test/{directory}')
        files_list = os.listdir(path)
        class_size[directory] = 0
        class_words[directory] = []
        for file in files_list:
            if file.endswith('.txt'):
                with open(os.path.join(path, file), 'r') as txt_file:
                    count += 1
                    count_all += 1
                    txt = txt_file.read().split()
                    class_size[directory] = class_size[directory] + len(txt)
                    class_words[directory] += txt
                    words_list.update(txt)
        class_number[directory] = count
        count = 0
    for key, value in class_words.items():
        word_count = {}
        for w in value:
            if w in word_count:
                word_count[w] += 1
            else:
                word_count[w] = 1
        count_of_words_in_class[key] = word_count
    # print(count_of_words_in_class)
    dictionary_path = os.path.join(project_dir, 'files/Classification-Train And Test/dictionary.txt')
    with open(dictionary_path, 'w') as dictionary_file:
        dictionary_file.write('\n'.join(words_list))
        dictionary_file.write(f'\n{len(words_list)}')
    # print(count_all)
    for key, value in class_number.items():
        class_number[key] = value / count_all

    prob_path = os.path.join(project_dir, 'files/Classification-Train And Test/classes_prob.txt')
    json_data = json.dumps(class_number, indent=2)
    with open(prob_path, 'w') as prob_file:
        prob_file.write(json_data)

    for key, value in class_words.items():
        result = os.path.join(project_dir, f'files/Classification-Train And Test/{key}.txt')
        json_data = json.dumps(value, indent=2)
        with open(result, 'w') as prob_file:
            prob_file.write(json_data)

    ###############################
    # print(class_words)
    # CLASS PROBS = CLASS_NUMBER[class_name]
    # DIC = WORDS
    # LEN_DIC = LEN(WORDS)
    # LEN_ CLASS = CLASS_SIZE[CLASS_NAME]
    classification = {}
    tests = 0
    correct_dir = 0
    v = len(words_list)
    for test_dir in classes:
        path = os.path.join(project_dir, f'files/Classification-Train And Test/{test_dir}/test')
        files_list = os.listdir(path)
        for file in files_list:
            tests += 1
            file_class = {}
            with open(os.path.join(path, file), 'r') as txt_file:
                txt = txt_file.read()
                words_of_class = txt.split()
            for selected_class in classes:
                prob = math.log(class_number[selected_class])
                count_class = class_size[selected_class]
                for word in words_of_class:
                    count_in_class = 1
                    count_in_class += count_of_words_in_class[selected_class].get(word, 0)
                    non_word = 0
                    if word not in words_list:
                        non_word = 1

                    # count_in_class = sum(w.count(word) for w in class_words[selected_class])
                    prob = prob + math.log((count_in_class / (count_class + v + non_word)))
                if selected_class == test_dir:
                    correct = 'yes'
                else:
                    correct = 'no'
                file_class[selected_class] = {'prob': prob, 'correct': correct}

            max_value = max(file_class.values(), key=lambda x: x['prob'])
            max_class = ''
            for key, value in file_class.items():
                if value == max_value:
                    max_class = key
                    break
            classification[file] = {max_class: max_value}
    # print(classification)

    result = os.path.join(project_dir, 'files/Classification-Train And Test/result.txt')
    json_data = json.dumps(classification, indent=2)
    with open(result, 'w') as prob_file:
        prob_file.write(json_data)

    for correct in classification.values():
        for c in correct.values():
            if c.get('correct') == 'yes':
                correct_dir += 1
    # accuracy
    print(correct_dir)
    print(tests)
    acc = (correct_dir / tests)
    print(acc)


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
        spell_correction()
        # spell_correction()
        # d = enchant.Dict("en_US")
        # print(d.check("enchant"))
        # print(min_edit_distance('acress','acres'))
        # x = min_edit_distance('acres', 'acress')
        # print(x)
        # channel_model(x)
    if base_menu == '3':
        classification_dictionary()
