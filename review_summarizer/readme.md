# Toy Amazon review summarizer + OpenAI's Completion Endpoint

Extremely simple script for trying out OpenAI's completion API with something
mildly useful. This requests ~10 reviews for each star rating for a product
from Amazon, packs it all into a prompt, and asks some questions.

## Usage
In a `.env` file, write:
```
OPENAI_API_KEY=<your API key>
```

Dependencies:
```
pip3 install requests
pip3 install bs4
pip3 install hyper  # no longer maintained but needed for http/2
pip3 install python-dotenv  # allows loading and reading .env files
pip3 install openai
```

Example:
```
$ python3 amazon_review_summarizer.py
ASIN of Amazon product:
0393541215
Extra questions (start with 3, <enter> without input to skip):
3. Would you read this book for fun?
Querying...

1. The top 5 pros of this product are:
-The author is very knowledgeable about the subject matter
-The book is very well organized
-The recipes are easy to follow
-The photos are helpful
-The book is reasonably priced

2. The top 5 cons of this product are:
-The photos are not of the highest quality
-Some of the recipes are too complicated for the average home cook
-There is too much information in the book
-The book is too long
-The author is too opinionated

3. I would not read this book for fun, but I would read it for information.
```

```
python3 amazon_review_summarizer.py
ASIN of Amazon product:
B08P4M89GP
Extra questions (start with 3, <enter> without input to skip):
3. what is this made of, and is it safe for storing food?
Querying Amazon...
Retrieving reviews from Amazon...
# 1-star reviews retrieved: 0
# 2-star reviews retrieved: 3
# 3-star reviews retrieved: 3
# 4-star reviews retrieved: 10
# 5-star reviews retrieved: 10
Total len of bad reviews: 1392
Total len of good reviews: 5869
------------------- REVIEW SUMMARY -------------------
1. The top 5 pros of this product are that they are cute, a good size, come with labels and a pen, have a shaker top, and are stackable.
2. The top 5 cons of this product are that the pen washes off easily, the shaker top is too small, the labels are not permanent, the jars are not dishwasher safe, and they are not microwave safe.
3. This product is made of glass and is safe for storing food.
------------------------------------------------------
```

## Some observations:
* Unsurprisingly, squeezing as many reviews into context is necessary for the model to return something useful.
* If we truncate reviews (not implemented in this version), keeping the end of a review yields far better results than keeping the beginning of a review. Possibly because we as humans tend to not get to the point until the end, reserving the start for context such as "I've used this item for 3 years".
* Sometimes the model returns bullet points and sometimes a text summarization.
* If the reviews are contradictory, the summary can be contradictory as well. In the jar example, the top 5 pros and top 5 cons both include stackability.
* Lower temperature tends to lead to less interesting responses.
* Adding more than one question also tends to lead the model to spit out 1 instead of 5 reasons.