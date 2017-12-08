import requests
from bs4 import BeautifulSoup
import re
import codecs
import time
import openpyxl
import pandas as pd

movieMessage=[]
movieMessage.append(['rank', 'movieName','movieShortShow', 'Introduction', 'director','mainActor',\
         'score', 'year', 'type', 'country', 'language' , 'review', 'starFive', 'starFour',\
         'starThree', 'starTwo', 'starOne'])
Download_url='https://movie.douban.com/top250'

def getHtml(url):
    headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36'}
    html=requests.get(url,headers=headers).content
    return html

def getMovieItem(movieHtml):
    movieSoup = BeautifulSoup(movieHtml)
    movieYear = movieSoup.find('span', attrs={'class': 'year'}).get_text().strip("()")
    movieMainMessage = movieSoup.find('div', attrs={'class': 'subjectwrap clearfix'})
    movieStarInfo = movieMainMessage.find('div', attrs={'id': 'interest_sectl'})
    movieInfo = movieMainMessage.find('div', attrs={'id': 'info'})
    messageList = movieInfo.get_text().strip().split('\n')
    movieDirector = messageList[0].split(':')[1] #导演
    movieActor = messageList[2].split(':')[1]   #主演
    movieType = messageList[3].split(':')[1]   #电影类型
    movieCountry = messageList[4].split(':')[1]   #电影制作国家
    movieLanguage = messageList[5].split(':')[1] #电影语言
    movieScore = movieStarInfo.find('strong', attrs={'class': 'll rating_num'}).get_text()  #电影评分
    movieReviewNum = movieStarInfo.find("span", attrs={'property': 'v:votes'}).get_text()  #电影评论
    movieScoreGrade = list(map(lambda x: x.get_text(), movieStarInfo.find_all('span', attrs={'class': 'rating_per'})))  #电影星级
    movieBriefIntroduction = movieSoup.find('span', attrs={'property': 'v:summary'}).get_text().strip().replace(' ','').replace('\u3000','') #电影剧情简介
    print(movieBriefIntroduction)
    return [movieBriefIntroduction,movieDirector,movieActor,movieScore,movieYear,movieType, \
            movieCountry,movieLanguage,movieReviewNum]+movieScoreGrade
def getmessage(html):
    soup=BeautifulSoup(html)
    movie_lists=soup.find('ol',attrs={'class':'grid_view'})
    for movieList in movie_lists.find_all('li'):
      try:
        moviePics=movieList.find('div',attrs={'class':'pic'})
        movieMainurls=moviePics.find('a',attrs={'href':re.compile('https:')})
        movieUrl=movieMainurls['href']
        movieNum=moviePics.find('em').text
        print(movieNum)  #电影排名
        movieInfo=movieList.find('div',attrs={'class':'info'})
        movieHd=movieInfo.find('div',attrs={'class':'hd'})
        movieName=movieHd.find('span',attrs={'class':'title'}).get_text()
        movieBd=movieList.find('div',attrs={'class':'bd'})
        if  movieBd.find('p',attrs={'class':'quote'}):
            movieShortShow= movieBd.find('p',attrs={'class':'quote'}).find('span',attrs={'class':'inq'}).string
        else :
            movieShortShow='无'
        movieHtml=getHtml(movieUrl)
        movieItemMes=getMovieItem(movieHtml)
        movieMessage.append([movieNum,movieName,movieShortShow]+movieItemMes)

      except Exception as e:
          print(e)
    next_page=soup.find('span',attrs={'class':'next'}).find('a')
    if next_page:
        return Download_url+next_page['href']
    return None

def main():
    url=Download_url
    while url:
        time.sleep(5)
        try:
            html=getHtml(url)
            url=getmessage(html)
        except Exception  as e:
            print(e)
    movie_data = pd.DataFrame(movieMessage)
    movie_data.to_excel(r'./dataSet/movieData.xlsx')
if __name__=='__main__':
    main()
