<h1> Janex: Ultimate Edition </h1>

I've realised that releasing multiple editions of Janex made things rather complicated for people to use the libraries, having to use a different library for a different method, also it's rather pointless me trying to maintain three or four individual libraries, so I've rewritten them to work inter-changeably and compiled them into this one singular library!

Please note this requires PyTorch, Spacy and NLTK to be installed, which will automatically install when using pip.

```bash
python3 -m pip install JanexUltimate
```

<h2> How to use </h2>

There are four flavours of Janex which are now bound into one library.

- Janex Python
- Janex PyTorch
- Janex Spacy
- Janex NLG

<h3> Janex Python </h3>

<h4> Tokenization </h4>

The tokenize(input_string) function tokenizes the input string into individual words, removing punctuation and converting all characters to lowercase.

Example usage:

```python
from janex.janexpython import *

input_string = "Hello, this is a sample sentence."
words = tokenize(input_string)
print(words)  # Output: ['hello', 'this', 'is', 'a', 'sample', 'sentence']
```

<h4> Word Stemming </h4>

The stem(input_word) function reduces a word to its base form by removing common suffixes.

Example usage:

```python
input_word = "running"
stemmed_word = stem(input_word)
print(stemmed_word)  # Output: "run"
```

<h4> String Vectorization </h4>

The string_vectorize(input_string) function converts a string into a numpy array of ASCII values representing each character.

Example usage:

```python
input_string = "hello"
vector = string_vectorize(input_string)
print(vector)  # Output: array([104, 101, 108, 108, 111])
```

<h4> Reshape Array Dimensions </h4>

The reshape_array_dimensions(array, dimensions) function reshapes the dimensions of a numpy array.

Example usage:

```python
import numpy as np

array = np.array([1, 2, 3, 4, 5, 6])
new_dimensions = (2, 3)
reshaped_array = reshape_array_dimensions(array, new_dimensions)
print(reshaped_array)  # Output: array([[1, 2, 3], [4, 5, 6]])
```

<h4> Cosine Similarity Calculation </h4>

The calculate_cosine_similarity(vector1, vector2) function calculates the cosine similarity between two numpy arrays.

Example usage:

```python
import numpy as np

vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])
similarity = calculate_cosine_similarity(vector1, vector2)
print(similarity)  # Output: 0.9746318461970762
```

<h4> Intent Classifier Toolkit </h4>

The IntentClassifier class provides functionality for intent classification based on pre-trained vectors and intents.

Example usage:

```python
classifier = IntentClassifier()
classifier.set_vectorsfp("vectors.json")
classifier.set_intentsfp("intents.json")
classifier.set_dimensions((300, 300))

classifier.train_vectors()

input_string = "How can I reset my password?"
intent = classifier.classify(input_string)
print(intent)  # Output: {'tag': 'password_reset', 'patterns': ['How can I reset my password?'], 'responses': ['You can reset your password by...']}
```

<h3> Janex PyTorch </h3>

The Janex PyTorch library provides tools for intent classification and response generation using PyTorch.

```
from JanexUltimate.janexpytorch import *
```

<h4> Initializing JanexPT </h4>

To initialize the JanexPT class, provide the file path to the intents JSON file.

```python
janex_pt = JanexPT(intents_file_path)
```

<h4> Setting device </h4>

You can set the device for PyTorch operations (e.g., "cpu" or "cuda") using the set_device method.

```python
janex_pt.set_device("cpu")
```

<h4> Comparing Patterns </h4>

To compare patterns and classify intents, use the pattern_compare method.

```python
intent = janex_pt.pattern_compare(input_string)
print(intent)
```

<h4> Modifying data path </h4>

```python
janex_pt.modify_data_path(new_path)
```

<h4> Training program </h4>

The library includes a training program that can be executed to train the model. Simply call the trainpt method.

```python
janex_pt.trainpt()
```

<h4> Example </h4>

Here's an example of how to use the Janex PyTorch library:

```python
from Janex_PyTorch import JanexPT

intents_file_path = "intents.json"
janex_pt = JanexPT(intents_file_path)

input_string = "How can I reset my password?"
intent = janex_pt.pattern_compare(input_string)
print(intent)

response = "Sample response"
new_response = janex_pt.generate_response_with_synonyms(response, strength=2)
print(new_response)
```

