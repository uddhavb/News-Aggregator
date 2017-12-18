

#input rss feeds
feeds = [
    'http://www.sfgate.com/rss/feed/Tech-News-449.php',
    'http://feeds.feedburner.com/TechCrunch/startups',
    'http://news.cnet.com/8300-1001_3-92.xml',
    'http://www.zdnet.com/news/rss.xml',
    'http://www.computerweekly.com/rss/Latest-IT-news.xml',
    'http://feeds.reuters.com/reuters/technologyNews',
    'http://www.tweaktown.com/news-feed/'
]

import feedparser
import nltk
from bs4 import BeautifulSoup
corpus = []
titles=[]
ct = -1
nkeywords=4  # number of keywords we will  use
for feed in feeds:
    d = feedparser.parse(feed)
    print d.status
    for e in d['entries']:
	   mySoup = BeautifulSoup(e['description'],'lxml')
	   soupyText = mySoup.get_text()
	   words= nltk.wordpunct_tokenize(soupyText)
	   lowerwords=[x.lower() for x in words if len(x) > 1] #make lower case. Also remove useless 1 character words like 'a','.' etc
	   ct += 1
	   print ct, "TITLE", e['title'].encode("utf-8")
	   #if len(lowerwords) >= nkeywords:
	   corpus.append(lowerwords)
	   titles.append(e['title'])

## define tf-idf
import math
import operator
def freq(word, document): return document.count(word)
def wordCount(document): return len(document)
def numDocsContaining(word,documentList):
  count = 0
  for document in documentList:
    if freq(word,document) > 0:
      count += 1
  return count
def tf(word, document): return (freq(word,document) / float(wordCount(document)))
def idf(word, documentList): return math.log(len(documentList) / numDocsContaining(word,documentList))
def tfidf(word, document, documentList): return (tf(word,document) * idf(word,documentList))


# extract keywords using tf-idf
def top_keywords(n,doc,corpus):
    d = {}
    for word in set(doc):
        d[word] = tfidf(word,doc,corpus)
    sorted_d = sorted(d.iteritems(), key=operator.itemgetter(1)) # you want to sort wrt the tfidf value and not the word, hence pass '1' to itemgetter
    sorted_d.reverse()
    return [w[0] for w in sorted_d[:n]]   
key_word_list=set()

[[key_word_list.add(x) for x in top_keywords(nkeywords,doc,corpus)] for doc in corpus]

print key_word_list

ct=-1
for doc in corpus:
   ct+=1
   print ct,"KEYWORDS"," ".join(top_keywords(nkeywords,doc,corpus)) #.encode("utf-8")

#compute the tf-idf for each term again
feature_vectors=[]
n=len(corpus)

for document in corpus:
    vec=[]
    [vec.append(tfidf(word, document, corpus) if word in document else 0) for word in key_word_list]
    feature_vectors.append(vec) 

# print feature_vectors
import numpy
mat = numpy.empty((n, n))
for i in xrange(0,n):
	for j in xrange(0,n):
		mat[i][j] = nltk.cluster.util.cosine_distance(feature_vectors[i],feature_vectors[j]) #calculating the cosine distance
		
# hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
t = 0.8
Z = linkage(mat, 'single')
d = dendrogram(Z, color_threshold=t)
from matplotlib import pyplot
print Z
import pylab
pylab.savefig( "dendo.png" ,dpi=800)

## extract data
def extract_clusters(Z,threshold,n):
   clusters={}
   ct=n
   for row in Z:
      if row[2] < threshold:
          n1=int(row[0])
          n2=int(row[1])

          if n1 >= n:
             l1=clusters[n1] 
             del(clusters[n1]) 
          else:
             l1= [n1]
      
          if n2 >= n:
             l2=clusters[n2] 
             del(clusters[n2]) 
          else:
             l2= [n2]    
          l1.extend(l2)  
          clusters[ct] = l1
          ct += 1
      else:
          return clusters

clusters = extract_clusters(Z,t,n)

for key in clusters:
   print "=============================================" 
   for id in clusters[key]:
       print id,titles[id]

