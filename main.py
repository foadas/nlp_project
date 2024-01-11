import json
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
                    if min_edit_distance(key, word) <= 1:
                        candidates_array.append(key)
                        break
            candidates[word] = candidates_array
    print(candidates)


def min_edit_distance(word1, word2):
    # truning 1 to 2
    len_word1 = len(word1)
    len_word2 = len(word2)

    # Initialize a matrix to store the minimum edit distances
    dp = [[0] * (len_word2 + 1) for _ in range(len_word1 + 1)]
    operations = [[""] * (len_word2 + 1) for _ in range(len_word1 + 1)]

    # Initialize the first row and column
    for i in range(len_word1 + 1):
        dp[i][0] = i
        operations[i][0] = 'D'
    for j in range(len_word2 + 1):
        dp[0][j] = j
        operations[0][j] = 'I'

    # Fill in the matrix based on minimum edit distances
    for i in range(1, len_word1 + 1):
        for j in range(1, len_word2 + 1):
            # Set uniform cost of 1 for all operations
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
    changes = {}

    while i > 0 or j > 0:
        operation = operations[i][j]
        # print(i, j, operation)
        if operation == "D":
            if i != 1:
                print(i)
                changes["insert"] = f'{word1[i - 2]}|{word1[i - 1]}'
            else:
                print(i)
                print(word1)
                changes["insert"] = f'{word1[i]}|{word1[i - 1]}'
            # .append(f"Delete '{word1[i - 1]}' at position {i}")
            i -= 1
        elif operation == "I":
            if j != 1:
                changes["delete"] = f'{word2[j - 2]}|{word2[j - 1]}'
                print(j)
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

    # The minimum edit distance is stored in the bottom-right cell of the matrix
    return changes


def channel_model(xw):
    dataset = read_file('files/SpellCorrection/test/Dictionary/Dataset.data').split(' ')

    if 'delete' in xw:

        w = xw['delete'].split('|')[1]
        x = xw['delete'].split('|')[0]
        matrix = read_file('files/SpellCorrection/test/Confusion Matrix/del-confusion.data').replace("'", '"')
        matrix = json.loads(matrix)
        matrix_value = matrix[f'{x + w}']
        count = 1
        # print(dataset)
        # dataset = ['technologies', 'esssss', 'fffesee', 'example', 'yes', 'guess']
        for word in dataset:
            count += word.count(w)
        print(count)
        print(matrix_value)

    elif 'insert' in xw:

        w = xw['insert'].split('|')[1]
        x = xw['insert'].split('|')[0]
        matrix = read_file('files/SpellCorrection/test/Confusion Matrix/ins-confusion.data').replace("'", '"')
        matrix = json.loads(matrix)
        count = 1
        print(x, w)
        matrix_value = matrix[f'{x + w}']
        for word in dataset:
            count += word.count(f'{x}')
        print(count)
        print(matrix_value)

    elif 'trans' in xw:

        w = xw['trans'][1]
        x = xw['trans'][0]
        matrix = read_file('files/SpellCorrection/test/Confusion Matrix/Transposition-confusion.data').replace("'", '"')
        matrix = json.loads(matrix)
        matrix_value = matrix[f'{x + w}']
        count = 1
        for word in dataset:
            count += word.count(f'{x + w}')
        print(count)
        print(matrix_value)

    elif 'sub' in xw:

        w = xw['sub'].split('|')[1]
        x = xw['sub'].split('|')[0]
        matrix = read_file('files/SpellCorrection/test/Confusion Matrix/sub-confusion.data').replace("'", '"')
        matrix = json.loads(matrix)
        matrix_value = matrix[f'{x + w}']
        count = 1
        for word in dataset:
            count += word.count(f'{w}')
        print(count)
        print(matrix_value)

    return '{:.18f}'.format(matrix_value / count)


def classification_dictionary():
    classes = ['Comp.graphics', 'rec.autos', 'sci.electronics', 'soc.religion.christian', 'talk.politics.mideast']
    words = set()
    class_words = {}
    count = 0
    count_all = 0
    for directory in classes:
        path = os.path.join(project_dir, f'files/Classification-Train And Test/{directory}')
        files_list = os.listdir(path)
        for file in files_list:
            if file.endswith('.txt'):
                with open(os.path.join(path, file), 'r') as txt_file:
                    count += 1
                    count_all += 1;
                    txt = txt_file.read().split()
                    words.update(txt)
        class_words[directory] = count
        count = 0
    dictionary_path = os.path.join(project_dir, 'files/Classification-Train And Test/dictionary.txt')
    with open(dictionary_path, 'w') as dictionary_file:
        dictionary_file.write('\n'.join(words))
        dictionary_file.write(f'\n{len(words)}')
    # print(count_all)
    for key, value in class_words.items():
        class_words[key] = value / count_all

    prob_path = os.path.join(project_dir, 'files/Classification-Train And Test/classes_prob.txt')
    json_data = json.dumps(class_words, indent=2)
    with open(prob_path, 'w') as prob_file:
        prob_file.write(json_data)

    ###############################

    for test_dir in classes:
        path = os.path.join(project_dir, f'files/Classification-Train And Test/{test_dir}/test')
        files_list = os.listdir(path)
        for file in files_list:
            words = file.split()
            for word in words:
                pass


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
            # print(min_edit_distance('acress','acres'))
            x = min_edit_distance('acress', 'acerss')
            print(x)
            channel_model(x)
    if base_menu == '3':
        classification_dictionary()
