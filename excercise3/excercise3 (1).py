import nltk
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer

# Step 1: Reading the Moby Dick file
nltk.download('gutenberg')  # Download the Gutenberg dataset
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Step 2: Tokenization
tokens = word_tokenize(moby_dick)

# Step 3: Stop-words filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Step 4: Parts-of-Speech (POS) tagging
pos_tags = pos_tag(filtered_tokens)

# Step 5: POS frequency
pos_counts = FreqDist(tag for word, tag in pos_tags)
top_pos = pos_counts.most_common(5)
print("Top 5 POS and their counts:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

# Step 6: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word, pos) for word, pos in pos_tags[:20]]
print("Lemmatized tokens:")
print(lemmas)

# Step 7: Plotting frequency distribution
pos_counts.plot(30, cumulative=False)
plt.title('POS Frequency Distribution')
plt.xlabel('POS')
plt.ylabel('Frequency')
plt.show()
