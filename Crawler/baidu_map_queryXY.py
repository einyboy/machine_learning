#-*-coding:utf-8-*-
import json
from urllib.request import urlopen, quote
import requests,csv
import pandas as pd #导入这些库后边都要用到

def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = '8f296a66c1b4d1794473b1a3dc42d92d'
    add = quote(address) #由于本文城市变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + add  + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode() #将其他编码的字符串解码成unicode
    temp = json.loads(res) #对json数据进行解析
    return temp

    
b = '百色新潮网吧'
c= 0#将第二列price读取出来并清除不需要字符
result = getlnglat(b)['result']
lng = result['location']['lng'] #采用构造的函数来获取经度
lat = result['location']['lat'] #获取纬度
precise = result['precise']
str_temp = '{"lat":' + str(lat) + ',"lng":' + str(lng) + ',"count":' + str(c) +'},'
print('lat:{} lng:{} precise:{}'.format(lng, lat, precise)) #也可以通过打印出来，把数据copy到百度热力地图api的相应位置上
print('{},{}'.format(lng, lat))