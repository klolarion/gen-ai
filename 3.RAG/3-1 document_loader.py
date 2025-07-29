import os

import bs4
from langchain_community.document_loaders import WebBaseLoader, TextLoader, DirectoryLoader, CSVLoader
from glob import glob


# # WebLoader
# 여러 개의 url 지정 가능
# url1 = "https://blog.langchain.dev/customers-replit/"
# url2 = "https://blog.langchain.dev/langgraph-v0-2/"
#
# loader_web = WebBaseLoader(
#     web_paths=(url1, url2),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("article-header", "article-content")
#         )
#     ),
# )
# docs = loader_web.load()
# len(docs)
#
# print(docs[0])

# # TextLoader
# loader_text = TextLoader("../history.txt", encoding="utf-8")
# data = loader_text.load()
#
# print(type(data))
# print(len(data))
#
# print(data)
# print(len(data[0].page_content))
# print(data[0].metadata)

# 커스텀 로더 불러오기 - encoding을 맞춰야 함
# class UTF8TextLoader(TextLoader):
#     def __init__(self, file_path):
#         super().__init__(file_path, encoding='utf-8')
#
# # DirectoryLoader
# files = glob(os.path.join('../', '*.txt'))
# print(files)
#
# loader = DirectoryLoader(path='../', glob='*.txt', loader_cls=UTF8TextLoader)
#
# data = loader.load()
#
# print(len(data))


# CSVLoader
loader = CSVLoader(file_path='../한국주택금융공사_주택금융관련_지수_20160101.csv', encoding='cp949')
data = loader.load()

print(len(data))
print(data[0])