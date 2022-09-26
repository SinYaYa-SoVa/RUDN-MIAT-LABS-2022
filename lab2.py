import glob
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import re


if __name__ == '__main__':
    data = []
    for path in glob.glob('./news_science/*'):
        texts = []
        for filename in tqdm(glob.glob(path + '/*.txt')):
            texts.append(open(filename, 'r').read().strip())

        data.append(pd.DataFrame({'text': texts}))
        data[-1]['genre'] = path.split('/')[-1][:3]  # 'new' или 'sci'

    data = pd.concat(data)
    print(data.sample(3))

    nltk.download('stopwords')
    nltk.download('punkt')

    wt = []
    for index, row in data.iterrows():
        wt.append(row['text'])
    wt = str(wt)


    stopwords = set(stopwords.words('russian'))
    word_tokens = word_tokenize(wt)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stopwords]
    for i in range(len(filtered_sentence)):
        filtered_sentence[i] = re.sub(r'\n|\\n', '', filtered_sentence[i])
    filtered_sentence = ' '.join([w for w in filtered_sentence])
    # sentence = nltk.sent_tokenize(filtered_sentence)
    # clean_text = re.sub(r'\n', '', filtered_sentence)
    # wordcloud = WordCloud().generate(clean_text)
    # print(clean_text)

    print(filtered_sentence)