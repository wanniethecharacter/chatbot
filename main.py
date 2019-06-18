import pandas as pd
import re
pd.set_option('display.max_colwidth', 200)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('data.csv')
convo = df.iloc[:, 0]


clist = []
def qa_pairs(x):
    cpairs = re.findall(": (.*?)(?:$|\n)", x)
    clist.extend(list(zip(cpairs, cpairs[1:])))
convo.map(qa_pairs);
convo_frame = pd.Series(dict(clist)).to_frame().reset_index()
convo_frame.columns = ['q', 'a']


vectorizer = TfidfVectorizer(ngram_range=(1,3))
vec = vectorizer.fit_transform(convo_frame['q'])

def get_response(q):
    my_q = vectorizer.transform([q])
    cs = cosine_similarity(my_q, vec)
    rs = pd.Series(cs[0]).sort_values(ascending=False)
    rsi = rs.index[0]
    return convo_frame.iloc[rsi]['a']



if __name__ == '__main__':
    print('Talk with me')
    while True:
        p = str(input())
        print(get_response(p))
