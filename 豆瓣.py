import requests
from bs4 import BeautifulSoup
import re
import codecs
import time
import openpyxl
import pandas as pd

movie_message=[]
movie_message.append(['电影名编号','电影名','导演','主演',\
                      '电影星级','电影评分','年份','类型','制片国家',\
                      '电影简介','评论数','热评1','热评2','热评3','热评4','热评5'])
Download_url='https://movie.douban.com/top250'

def gethtml(url):
    headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36'}
    html=requests.get(url,headers=headers).content
    return html
def getmessage(html):
    soup=BeautifulSoup(html)
    movie_lists=soup.find('ol',attrs={'class':'grid_view'})
    for movie_list in movie_lists.find_all('li'):
      try:
        movie_pics=movie_list.find('div',attrs={'class':'pic'})
        movie_mainurls=movie_pics.find('a',attrs={'href':re.compile('https:')})
        movie_url=movie_mainurls['href']
        movie_num=movie_pics.find('em').text
        print(movie_num)
        movie_info=movie_list.find('div',attrs={'class':'info'})
        movie_hd=movie_info.find('div',attrs={'class':'hd'})
        movie_name=movie_hd.find('span',attrs={'class':'title'}).text
        movie_bd=movie_list.find('div',attrs={'class':'bd'})
        if  movie_bd.find('p',attrs={'class':'quote'}):
            movie_shortshow= movie_bd.find('p',attrs={'class':'quote'}).find('span',attrs={'class':'inq'}).string
        else :
            movie_shortshow='无'
        time.sleep(5)
        movie_html=gethtml(movie_url)
        movie_soup=BeautifulSoup(movie_html)
        movie_content=movie_soup.find('div',attrs={'id':'content'})
        movie_year=movie_soup.find('span',attrs={'class':'year'}).get_text().strip("()")
        movie_article=movie_content.find('div',attrs={'class':'article'})
        movie_main_message=movie_article.find('div',attrs={'class':'subjectwrap clearfix'})
        movie_star_info=movie_main_message.find('div',attrs={'id':'interest_sectl'})
        movie_info=movie_main_message.find('div',attrs={'id':'info'})
        message_list=movie_info.get_text().strip().split('\n')
        movie_director=message_list[0].split(':')[1]
        movie_actor=message_list[2].split(':')[1]
        movie_type=message_list[3].split(':')[1]
        movie_area=message_list[4].split(':')[1]
        movie_language=message_list[5].split(':')[1]
        movie_score=movie_star_info.find('strong',attrs={'class':'ll rating_num'}).get_text()
        movie_review_num=movie_star_info.find("span",attrs={'property':'v:votes'}).get_text()
        movie_score_grade=list(map(lambda x:x.get_text(),movie_star_info.find_all('span',attrs={'class':'rating_per'})))
        movie_short_show=movie_soup.find('span',attrs={'property':'v:summary'}).get_text()
        movie_message.append([movie_num,movie_name,movie_director,movie_actor,movie_score,movie_year,movie_type, \
                              movie_area,movie_language,movie_short_show,movie_review_num]+movie_score_grade)
      except Exception as e:
          print(e)
    next_page=soup.find('span',attrs={'class':'next'}).find('a')
    if next_page:
        return (movie_message,Download_url+next_page['href'])
    return (movie_message,None)

def main():
    url=Download_url
    while url:
        time.sleep(5)
        try:
            html=gethtml(url)
            message,url=getmessage(html)
            pd.DataFrame(message).to_excel('data1.xlsx')
        except Exception  as e:
            print(e)
    hjw = pd.DataFrame(movie_message)
    hjw.to_excel('data.xlsx')
if __name__=='__main__':
    main()
