from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#load_dotenv

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


documents=[
    'Virat kohli is greatest batsman in India now a days',
    'Bumrah is best bowler in India',
    'Shane watson is excellent all-rounder from Australia',
    'Shane Warne is best spinner so far in Australia'
]

query=input('ask basic question about cricket about players:')

doc_embedding=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

scores=cosine_similarity([query_embedding],doc_embedding)[0] # query embedding has to be in 2d list
# we need to get only 1d list and we will store that in the score variable
# also we need to sort it and we also need to make sure order should not change, so we will use enumarate function
print(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]) # we have sorted in ascending order and extracted highest (last one)
# once we extract the highest one we will retrieve corresponding answer
index,answer=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
full_answer=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print('using full_answer method: ',documents[full_answer[0]])
print('using index-answer method: ',documents[index])
