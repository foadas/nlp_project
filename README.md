<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Natural Language Processing (NLP) Project</h1>

<p>This project contains modules for text preprocessing, spell correction, and text classification.</p>

<h2>Features</h2>

<h3>1. Preprocessing</h3>
<ul>
    <li><strong>Tokenization:</strong> Splits the text into individual words or tokens.</li>
    <li><strong>Lowercase Folding:</strong> Converts all text characters to lowercase.</li>
    <li><strong>Word Count:</strong> Counts the frequency of each word in the text.</li>
    <li><strong>Stemming:</strong> Reduces words to their base form using the Porter Stemmer.</li>
</ul>

<h3>2. Spell Correction</h3>
<ul>
    <li>Performs spell correction using language and channel models.</li>
    <li><strong>Confusion Matrices:</strong> Uses deletion, insertion, substitution, and transposition matrices for error correction.</li>
    <li><strong>Edit Distance:</strong> Finds the closest correct word by calculating the minimum edit distance.</li>
    <li><strong>Dictionary-Based Suggestions:</strong> Provides word suggestions from a dictionary and selects the best option based on probabilities.</li>
</ul>

<h3>3. Text Classification</h3>
<ul>
    <li>Classifies documents into predefined categories using Naive Bayes.</li>
    <li><strong>Training Data:</strong> Trains on categories like `Comp.graphics`, `rec.autos`, and more.</li>
    <li><strong>Dictionary Creation:</strong> Builds a dictionary from training data and calculates word frequencies per class.</li>
    <li><strong>Prediction:</strong> Classifies test documents by computing the likelihood of words for each class.</li>
</ul>

</body>
</html>
