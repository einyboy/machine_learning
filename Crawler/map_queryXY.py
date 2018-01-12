#-*-coding:utf-8-*-
import json
import numpy as np
import math
import time
import jieba
import jieba.posseg as psg
import pandas as pd
from urllib.request import urlopen, quote
'''
利用高德地图api实现地址和经纬度的转换
'''
import requests

key = '8f296a66c1b4d1794473b1a3dc42d92d'  # 这里填写你的百度开放平台的key
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率


def baidu_getlnglat(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = '8f296a66c1b4d1794473b1a3dc42d92d'
    add = quote(address) #由于本文城市变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + add  + '&output=' + output + '&ak=' + ak + '&city=451000'
    req = urlopen(uri)
    res = req.read().decode() #将其他编码的字符串解码成unicode
    temp = json.loads(res) #对json数据进行解析
    #print(temp)
    result = temp['result']
    lng = result['location']['lng'] #采用构造的函数来获取经度
    lat = result['location']['lat'] #获取纬度
    return float(lng),float(lat)
    #return 0, 0
    
def gaode_geocode0(address):
    """
    利用百度geocoding服务解析地址获取位置坐标
    :param address:需要解析的地址
    :return:
    """
    geocoding = {'s': 'rsv3',
                 'key': key,
                 'city': '全国',
                 'address': address}
    res = requests.get(
        "http://restapi.amap.com/v3/geocode/geo", params=geocoding)
    if res.status_code == 200:
        json = res.json()
        status = json.get('status')
        count = json.get('count')
        if status == '1' and int(count) >= 1:
            geocodes = json.get('geocodes')[0]
            print(geocodes)
            lng = float(geocodes.get('location').split(',')[0])
            lat = float(geocodes.get('location').split(',')[1])
            return lng, lat
        else:
            print('================')
            return 0,0
    else:
        return  0,0
        
def gaode_geocode(address):
    parameters = {'address': address, 'key': 'cb649a25c1f81c1451adbeca73623251'}
    base = 'http://restapi.amap.com/v3/geocode/geo'
    response = requests.get(base, parameters)
    answer = response.json()
    print(answer)
    #lon, lat = answer['geocodes'][0]['location'].split(',')
    #print(address + "的经纬度：{},{}".format(lat, lon))
    #return float(lon),float(lat)
    return 0,0
    

    
def gcj02tobd09(lng, lat):
    """
    火星坐标系(GCJ-02)转百度坐标系(BD-09)
    谷歌、高德——>百度
    :param lng:火星坐标经度
    :param lat:火星坐标纬度
    :return:
    """
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09togcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)
    百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:转换后的坐标列表形式
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]


def wgs84togcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if out_of_china(lng, lat):  # 判断是否在国内
        return lng, lat
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    print(type(dlat),type(dlng))
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def gcj02towgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
        0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
        0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret
    
def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    if lng < 72.004 or lng > 137.8347:
        return True
    if lat < 0.8293 or lat > 55.8271:
        return True
    return False

def degree(degree = 0, minute = 0, second = 0):
    degree = 1.0*degree + minute/60. + second/3600.
    return degree
    
def test1():
    #address = input("请输入地址:")
    lng, lat = 106.62523831632139, 23.895816732806065
    address = '百色公安局'
    address = '乐业饭店'
    print('==================================')
    result1 = lng,lat = baidu_getlnglat(address)
    result2 = lng,lat = bd09togcj02(lng,lat)
    print(result1, result2)
    lng,lat =gaode_geocode(address)
    print(lng,lat)
    result4 = lng,lat = gcj02tobd09(lng,lat)
    #lng,lat = result3
    result5 = lng,lat = gcj02towgs84(lng, lat)
    #http://api.map.baidu.com/ag/coord/convert?from=0&to=4&x=106.62181661021562&y=23.89871131073383
    lng,lat =wgs84togcj02(lng,lat)
    result6 = gcj02tobd09(lng,lat)
    #print(result3, result4, result5)
    print(result6)
    
def test2():
    lng = degree(106,37,49)
    lat = degree(23,53,5)
    lng,lat =wgs84togcj02(lng,lat)
    result6 = gcj02tobd09(lng,lat)
    print('{},{}'.format(lng, lat))

def test3():
    filename = 'LG.xls'
    output = 'LG_XY.xls'
    df = pd.read_excel(filename, dtype=str)
    for idx, row in df.iterrows():
        try:
            address = row[ '酒店名称'];
            flng,flat = gaode_geocode(address)
            row['高德X'] = flng
            row['高德Y'] = flat
            lng,lat = gcj02tobd09(flng,flat)
            row['百度X'] = lng
            row['百度Y'] = lat
            
            lng,lat = gcj02towgs84(flng, flat)
            row['经度X'] = lng
            row['纬度Y'] = lat
            print('{} lng:{} lat:{}'.format(address, lng, lat))
        except:
            print('===================>{}'.format(row[ '酒店名称']))
    df.to_excel(output, encoding='gb2312')
def test4():
    filename = 'WB.xls'
    output = 'WB_XY.xls'
    df = pd.read_excel(filename, dtype=str)
    for idx, row in df.iterrows():
        try:
            address = row['名称'];
            flng,flat = baidu_getlnglat(address)
            row['百度X'] = flng
            row['百度Y'] = flat
            lng, lat = bd09togcj02(flng, flat)
            row['高德X'] = lng
            row['高德Y'] = lat          
            
            lng,lat = gcj02towgs84(flng, flat)
            row['经度X'] = lng
            row['纬度Y'] = lat
            print('{} lng:{} lat:{}'.format(address, lng, lat))
        except:
            print('===================>{}'.format(row[ '地址']))
    df.to_excel(output, encoding='gb2312')
def test5():
    address='百色公安局'
    gaode_geocode2(address)

def getPOSWord(word):
    keyword = []
    for x in psg.cut(s):
        if x.flag == 'ns' or x.flag == 'm':
            keyword.append(x.word)
    
    return ''.join(keyword)
    
def decodebdXY(x,y):
    url_format = "http://api.map.baidu.com/geoconv/v1/?coords={},{}&from=6&to=5&ak={}"
    url = url_format.format(x/100.0, y/100.0, key)
    print(url)
    lat = ''
    lnt = ''
    try:
        response = requests.get(url)
        pos = response.json()
        print(pos)
        
        if pos['status']==0:
            lnt, lat = pos['result'][0]['x'], pos['result'][0]['y']
    except:
        print("{},{} 转换失败".format(lnt,lat))
        lnt,lat = '',''
    return lnt,lat
    
def getPOI(keyword):
    cityCode='203'
    keyword = quote(keyword)
    cnt = 0
    rn = 10
    pn = 0
    rec = []
    for pn in range(2):
        url_format = 'http://api.map.baidu.com/?qt=s&c={}&wd={}&rn={}&pn={}&ie=utf-8&oue=1&fromproduct=jsapi&res=api&callback=BMap._rd._cbk{}&ak=E4805d16520de693a3fe707cdc962045'
        callback = np.random.randint(60000,90000)
        url = url_format.format(cityCode, keyword, rn, pn, callback)
        #print(url)
        #return
        #add = quote(address) #由于本文城市变量为中文，为防止乱码，先用quote进行编码
        try:
            req = urlopen(url, timeout=6)
            res = req.read().decode() #将其他编码的字符串解码成unicode
            res = res.replace('/**/BMap._rd._cbk{} && BMap._rd._cbk{}('.format(callback,callback),'')[:-1]
            #print(res)
            #return
            result = json.loads(res) #对json数据进行解析
            
            if 'content' not in result:
                continue
            content = result['content']
            for ent in content:
                cnt = cnt + 1
                addr = ent['addr']
                name = ent['name']
                lnt = ent['x']
                lat = ent['y']
                if '网' in name:
                    rec.append([name, addr, lnt, lat])
                    print('name:{} addr:{} {},{}'.format(name, addr, lnt, lat))
                    break
            
        except:
            print('==================pn:{}'.format(pn))
        time.sleep(6)
    return rec
    
def test6():
    filename = 'WB.xls'
    output = 'WB_XY.xls'
    wb_list = []
    df = pd.read_excel(filename, dtype=str)
    for idx, row in df.iterrows():
        try:
            keyword = row['名称'];
            d = df[df['名称']==keyword]
            count = len(d)
            if count == 1:
                rec = getPOI(keyword)
            else:
                keyword = row['地址']
                keyword = getPOSWord(keyword)
                rec = getPOI(keyword)
            if rec == []:
                keyword = row['地址']
                keyword = getPOSWord(keyword)
                rec = getPOI(keyword)
                
            for name, addr, lnt, lat in rec:
                wb_list.append([row['代码'], row['名称'], name, addr, lnt, lat])
                print('code:{} {}'.format(row['代码'],rec))
        except:
            print('===================>{}'.format(row[ '地址']))
    df = pd.DataFrame(wb_list, columns=['代码','名称', '网吧','地址','经度', '纬度'])
    df.to_excel('{}.xls'.format(keyword),  encoding='gb2312')

def test7():
    filename = 'WB_XY.xls'
    output = 'WB_XY_INFO.xls'
    wb_list = []
    df = pd.read_excel(filename, dtype=str)
    for idx, row in df.iterrows():
        lnt = row['经度']
        lat = row['纬度']
        lnt,lat = decodebdXY(float(lnt),float(lat))
        row['百度X'] = lnt
        row['百度Y'] = lat
        print('name:{} {},{}'.format(row['名称'], lnt, lat))
       
        lnt, lat = bd09togcj02(float(lnt), float(lat))
        row['高德X'] = lnt
        row['高德Y'] = lat
        #print('name:{} {},{}'.format(row['名称'], lnt, lat))
        lng,lat = gcj02towgs84(lnt, lat)
        row['经度X'] = lnt
        row['纬度Y'] = lat
        wb_list.append([])
        #print('name:{} {},{}'.format(row['名称'], lnt, lat))
        #df.to_excel(output, encoding='gb2312')
        #return
    df.to_excel(output, encoding='gb2312')
    
if __name__=='__main__':
    #test1();
    #test2();
    #test3()
    #test4()
    #test5()
    test6()
    test7()