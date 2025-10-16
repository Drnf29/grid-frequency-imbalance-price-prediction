# FinBERT imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# News imports
from GoogleNews import GoogleNews
import snscrape.modules.twitter as sntwitter
from psaw import PushshiftAPI

tickers = ["TSLA", "AAPL", "NVDA", "INTC", "PFE", "AMZN", "MSFT", "GOOGL"]

# FinBERT NLP model used to analyse financial text
model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[torch.argmax(probs)]
    return sentiment, probs.detach().numpy()[0]

# Google search
google_news = GoogleNews(lang='en', region='US')
google_news.search('Facebook stock')
results = google_news.results()
for r in results:
    print(r['title'])

print("----------------------------")

# Twitter search
#

print("----------------------------")

# Reddit posts
api = PushshiftAPI()
for post in api.search_submissions(after=start_time, subreddit='stocks', q='AAPL', filter=['title', 'url'], limit=5):
    print(post.title)
