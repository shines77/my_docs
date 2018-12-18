
http://www.huya.com/cache1min.php?m=chatMessage&tid=90126358&sid=90126358

参数：

```json
{
  "m": "chatMessage",
  "tid": 90126358,
  "sid": 90126358
}
```

响应：（获取聊天纪录历史）

```json
{
  "status": 1000,
  "time": 1544951831,
  "result": {
    "chats": []
  }
}
```


http://www.huya.com/udb_web/checkLogin.php

参数：（无）

响应：（登陆信息）

```json
{
  "isLogined": true,
  "userName": "wokss55",
  "uid": "497174994",
  "userNick": "\u4e91\u4fe1",
  "userLogo": null,
  "isHuya": true
}
```

```json
{
  "isLogined": true,
  "userName": "wokss55",
  "uid": 497174994,
  "userNick": "云信",
  "userLogo": null,
  "isHuya": true
}
```

http://user.huya.com/user/getUserInfo?callback=jQuery111103441958053755604_1544951829750&uid=497174994

参数：

callback: jQuery111103441958053755604_1544951829750
uid: 497174994

```json
{
    "message":"请求成功", "data":{"uid":497174994, "avatar":"https://huyaimg.msstatic.com/avatar/1003/a3/2ec1b37d52d1dc914763afce46b587_180_135.jpg?1543817092"}, "code":200
}
```

响应：（获取用户信息，uid虎牙号，头像地址）

```json
{
  "message": "请求成功",
  "data": {
    "uid": 497174994,
    "avatar": "https://huyaimg.msstatic.com/avatar/1003/a3/2ec1b37d52d1dc914763afce46b587_180_135.jpg?1543817092",
    "code": 200
  }
}
```

获取视频 (时间戳)

http://metric.huya.com/?ts=1544951833508

```json
{
    "code":0, "msg":"OK"
}
```

http://api.huya.com/subscribe/getSubscribeStatus?callback=jQuery111103441958053755604_1544951829750&from_key=497174994&from_type=1&to_key=1774581086&to_type=2&_=1544951829751

参数：

```json
{
  "callback": "jQuery111103441958053755604_1544951829750",
  "from_key": 497174994,
  "from_type": 1,
  "to_key": 1774581086,
  "to_type": 2,
  "_": 1544951829751
}
```

响应：（订阅量）

```json
{
  "result": 0,
  "status": 1,
  "subscribe_count": 426816
}
```


http://www.huya.com/cache5min.php?m=WeekRank&do=getItemsByPid&pid=1774581086

参数：

```json
{
  "m": "WeekRank",
  "do": "getItemsByPid",
  "pid": 1774581086,
}
```

响应：(获取周贡榜列表)

```json
{
    error: false,
    code: 0,
    data: {
        vWeekRankItem: [
        0	{…}
            lUid	2192054090
            sNickName	菜泡饭
            iScore	1461500
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1068/2a/4eda0f91876c3e2cf01c9ef54e8ed3_180_135.jpg?1544303092
            iUserLevel	5
        1	{…}
            lUid	1453814503
            sNickName	遗忘时间
            iScore	1000000
            iGuardLevel	-1
            iNobleLevel	5
            sLogo	http://downhdlogo.yy.com/hdlogo/144144/3/45/81/1453814503/u1453814503kx70c9C.png
            iUserLevel	9
        2	{…}
            lUid	2280652606
            sNickName	宝贝黛妮
            iScore	966600
            iGuardLevel	-1
            iNobleLevel	6
            sLogo	https://huyaimg.msstatic.com/avatar/1004/fe/23b188e487e66c052f7b315825e143_180_135.jpg?1544577796
            iUserLevel	14
        3	{…}
            lUid	1790788337
            sNickName	眺望伯纳乌
            iScore	758000
            iGuardLevel	-1
            iNobleLevel	2
            sLogo	https://huyaimg.msstatic.com/avatar/1089/08/9bc8107fc151a2c6707629f2a7cb99_180_135.jpg?1540608386
            iUserLevel	18
        4	{…}
            lUid	2241310128
            sNickName	🐰花火
            iScore	507500
            iGuardLevel	-1
            iNobleLevel	2
            sLogo	https://huyaimg.msstatic.com/avatar/1017/32/d5879d93030896ec4baa27b3c14bc0_180_135.jpg?1544348968
            iUserLevel	10
        5	{…}
            lUid	2305884846
            sNickName	王总的陈秘书、007
            iScore	500000
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1092/fd/cb2a4f373166c4fd3de28c7eaad674_180_135.jpg?1533813560
            iUserLevel	10
        6	{…}
            lUid	1795633308
            sNickName	囩 飛 揚
            iScore	399000
            iGuardLevel	-1
            iNobleLevel	3
            sLogo	http://huyaimg.msstatic.com/avatar/1089/d2/86041062447a1a51e683b631e2580a_180_135.jpg
            iUserLevel	4
        7	{…}
            lUid	86057704
            sNickName	白薇薇家的守护丷夜殇
            iScore	366000
            iGuardLevel	-1
            iNobleLevel	5
            sLogo	https://huyaimg.msstatic.com/avatar/1090/57/6bbbe63af9dc1b628efde103a229d0_180_135.jpg?1532207904
            iUserLevel	15
        8	{…}
            lUid	2297263912
            sNickName	leftsun007
            iScore	351300
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1081/71/ccee1b6d6f5b946c896f7c0c05da84_180_135.jpg
            iUserLevel	9
        9	{…}
            lUid	2230342791
            sNickName	张自力
            iScore	349600
            iGuardLevel	-1
            iNobleLevel	4
            sLogo	https://huyaimg.msstatic.com/avatar/1044/5d/1b24844d57f3db1495f61859fdad84_180_135.jpg?1543707871
            iUserLevel	18
        10	{…}
            lUid	1800420257
            sNickName	星空中最低调的星❤007
            iScore	251900
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	http://huyaimg.msstatic.com/avatar/1090/2d/bcfb6cc4a6099032bd6ad37f790c25_180_135.jpg?1511116881
            iUserLevel	20
        11	{…}
            lUid	1215610876
            sNickName	妃子大婶
            iScore	201800
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	http://huyaimg.msstatic.com/avatar/1018/45/c3f725c86d8758f63d79c74f18dd3f_180_135.jpg?1522106039
            iUserLevel	9
        12	{…}
            lUid	121366295
            sNickName	乱乱
            iScore	151400
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	
            iUserLevel	3
        13	{…}
            lUid	2336580614
            sNickName	血祭夜
            iScore	146800
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1055/44/8f8d3732c8c19fa79bf7f429cfcf53_180_135.jpg?1543069259
            iUserLevel	8
        14	{…}
            lUid	1826096278
            sNickName	✨浅笑の母老虎✨🌙
            iScore	132000
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1093/36/2f43557be3c48cdb2ee0d476902419_180_135.jpg?1526427863
            iUserLevel	15
        15	{…}
            lUid	84584019
            sNickName	爸爸池
            iScore	131600
            iGuardLevel	-1
            iNobleLevel	2
            sLogo	https://huyaimg.msstatic.com/avatar/1090/f8/e904a947a3b17b572f3a3b69ccb36c_180_135.jpg?1537696200
            iUserLevel	13
        16	{…}
            lUid	2183210913
            sNickName	狼人杀-可达鸭
            iScore	130000
            iGuardLevel	-1
            iNobleLevel	2
            sLogo	https://huyaimg.msstatic.com/avatar/1016/64/d1cb4dda6bb2f37623d323b46ac617_180_135.jpg
            iUserLevel	7
        17	{…}
            lUid	153106347
            sNickName	意识本源
            iScore	114500
            iGuardLevel	-1
            iNobleLevel	4
            sLogo	http://huyaimg.msstatic.com/avatar/1049/6e/76f7944c0cf45b68f49161dfa7c542_180_135.jpg?1513010645
            iUserLevel	21
        18	{…}
            lUid	172548778
            sNickName	请叫我彬少
            iScore	109000
            iGuardLevel	1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1094/af/c42a64d9feedc427f8f3d61a7a33cb_180_135.jpg?1542377256
            iUserLevel	9
        19	{…}
            lUid	1828822412
            sNickName	大胖冉
            iScore	103100
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1090/b1/ad8b659e07ba96f5e3ba915769be24_180_135.jpg?1544552145
            iUserLevel	10
        20	{…}
            lUid	395304918
            sNickName	甜蜜轻语
            iScore	101000
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1047/cb/e1c939b87bc28c5f20d1bed6a33413_180_135.jpg?1527497931
            iUserLevel	20
        21	{…}
            lUid	2246306958
            sNickName	顾九
            iScore	100800
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	
            iUserLevel	4
        22	{…}
            lUid	1857552782
            sNickName	老丁
            iScore	99000
            iGuardLevel	-1
            iNobleLevel	2
            sLogo	https://huyaimg.msstatic.com/avatar/1083/55/4c593ea68aed6a7b936f0957520d91_180_135.jpg?1542981146
            iUserLevel	14
        23	{…}
            lUid	1760738204
            sNickName	Amber
            iScore	99000
            iGuardLevel	-1
            iNobleLevel	4
            sLogo	http://huyaimg.msstatic.com/avatar/1028/27/f8164b44402d0a75bd80437504ddfc_180_135.jpg?1500786174
            iUserLevel	17
        24	{…}
            lUid	675688343
            sNickName	小布咔丶007
            iScore	98400
            iGuardLevel	12
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1057/9b/7b301f64e2fdb030b4139b4227d8f3_180_135.jpg?1531998024
            iUserLevel	3
        25	{…}
            lUid	855143
            sNickName	日月先生
            iScore	98100
            iGuardLevel	1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1017/ce/3950960501cc2e69db9b1869023e33_180_135.jpg?1544452566
            iUserLevel	3
        26	{…}
            lUid	2328728555
            sNickName	爱骗豆的主播
            iScore	94000
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1000/52/fc81ca0c678c704a0b3620d02172c5_180_135.jpg?1544028770
            iUserLevel	8
        27	{…}
            lUid	2221997369
            sNickName	好@
            iScore	93800
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	http://huyaimg.msstatic.com/avatar/1097/16/12ed5f2b471bbdd5120b525b252cb3_180_135.jpg
            iUserLevel	4
        28	{…}
            lUid	1839275582
            sNickName	糯米✔️
            iScore	88000
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1048/81/2890041471644ef3e1266e40e7c276_180_135.jpg?1534107293
            iUserLevel	11
        29	{…}
            lUid	1199511957173
            sNickName	閒雲千里
            iScore	87700
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1021/3b/8015dcf44e849c0e6d196a8f2aa3ed_180_135.jpg?1543316216
            iUserLevel	7
        30	{…}
            lUid	1852183116
            sNickName	卡夫卡
            iScore	79200
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	http://huyaimg.msstatic.com/avatar/1078/5d/0808e4c731fe759dae5f97827291d6_180_135.jpg
            iUserLevel	12
        31	{…}
            lUid	1512337925
            sNickName	玩玩而已
            iScore	66200
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1069/80/2010dd3003c63802269e482ba10992_180_135.jpg?1544942212
            iUserLevel	10
        32	{…}
            lUid	1621306730
            sNickName	淹不死的小强～
            iScore	66000
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	http://huyaimg.msstatic.com/avatar/1072/60/e0d2c51767c77597f3e7c03f8f07f9_180_135.jpg
            iUserLevel	4
        33	{…}
            lUid	2354144385
            sNickName	碧玉
            iScore	61000
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1066/15/9cda6e746ed18b3bed2fd7efe04436_180_135.jpg?1544900349
            iUserLevel	2
        34	{…}
            lUid	2288097566
            sNickName	兔喵酱💭
            iScore	60400
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1072/a6/fb42432bfa2f987ecc08b093241594_180_135.jpg?1530782811
            iUserLevel	2
        35	{…}
            lUid	1753870610
            sNickName	小鱼儿与花无缺
            iScore	56800
            iGuardLevel	-1
            iNobleLevel	2
            sLogo	https://huyaimg.msstatic.com/avatar/1058/5c/6530209b89c71c5903dfa0785540c2_180_135.jpg?1542547125
            iUserLevel	12
        36	{…}
            lUid	71220572
            sNickName	江小白
            iScore	55000
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1081/39/c67f9285142f09391e6b2097efcd11_180_135.jpg?1528555759
            iUserLevel	11
        37	{…}
            lUid	1199512289995
            sNickName	林恩啊林恩
            iScore	54300
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1000/1e/e1bcc56503f31b24dc6757524a9597_180_135.jpg?1543604136
            iUserLevel	6
        38	{…}
            lUid	2331740740
            sNickName	华哥
            iScore	51200
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1002/6c/d165e988afb3eff8c46721c40b3c92_180_135.jpg?1539953018
            iUserLevel	7
        39	{…}
            lUid	1830454546
            sNickName	Edin
            iScore	48300
            iGuardLevel	-1
            iNobleLevel	2
            sLogo	http://huyaimg.msstatic.com/avatar/1016/4a/dc2a080994b08aa0443a746382302b_180_135.jpg
            iUserLevel	3
        40	{…}
            lUid	513835362
            sNickName	静伤
            iScore	46200
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	http://huyaimg.msstatic.com/avatar/1071/b8/1ff47dc9a0832a649825d0ecfeac40_180_135.jpg?1502805651
            iUserLevel	5
        41	{…}
            lUid	1199512603198
            sNickName	情之所深是你
            iScore	44600
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1013/b0/f08f91cc7e20956a5ac9be6e418356_180_135.jpg?1544882372
            iUserLevel	2
        42	{…}
            lUid	2320304646
            sNickName	玲児
            iScore	42600
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1014/9a/7e2ca2c53876cca606505820a1c086_180_135.jpg?1544886648
            iUserLevel	5
        43	{…}
            lUid	2339065281
            sNickName	我是一颗小JY
            iScore	37700
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	https://huyaimg.msstatic.com/avatar/1024/0d/2b06c604cd0677d69edbda456c8d1c_180_135.jpg
            iUserLevel	4
        44	{…}
            lUid	2186360705
            sNickName	从小就很抠门
            iScore	37600
            iGuardLevel	-1
            iNobleLevel	4
            sLogo	https://huyaimg.msstatic.com/avatar/1022/71/1170af463c3d1803aa0c812f87272d_180_135.jpg?1544197148
            iUserLevel	14
        45	{…}
            lUid	2252665268
            sNickName	企鹅崽
            iScore	35000
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	http://huyaimg.msstatic.com/avatar/1078/40/8ed36610a9012151ebad3693ced28a_180_135.jpg
            iUserLevel	2
        46	{…}
            lUid	127928908
            sNickName	奢望
            iScore	34400
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	
            iUserLevel	5
        47	{…}
            lUid	433955821
            sNickName	冷泪、007
            iScore	33600
            iGuardLevel	-1
            iNobleLevel	1
            sLogo	https://huyaimg.msstatic.com/avatar/1091/df/430d3a637343414ed2622b57ccd6f0_180_135.jpg?1544791105
            iUserLevel	14
        48	{…}
            lUid	2195338532
            sNickName	小璐子🍀
            iScore	31400
            iGuardLevel	-1
            iNobleLevel	-1
            sLogo	http://huyaimg.msstatic.com/avatar/1035/91/8acc37910a839430985f627e248f19_180_135.jpg?1514111054
            iUserLevel	5
        49: {
            lUid: 413081565
            sNickName: 11
            iScore: 30400
            iGuardLevel: -1
            iNobleLevel: -1
            sLogo	
            iUserLevel: 2
        }
        ],
        iStart: 1,
        iCount: 50,
        iTotal: 333,
    }
    "ts": 1544951416
}
```

payload:

```json
{
"error":false,"code":0,"data":{"vWeekRankItem":[{"lUid":2192054090,"sNickName":"\u83dc\u6ce1\u996d","iScore":"1461500","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1068\/2a\/4eda0f91876c3e2cf01c9ef54e8ed3_180_135.jpg?1544303092","iUserLevel":5},{"lUid":"1453814503","sNickName":"\u9057\u5fd8\u65f6\u95f4","iScore":"1000000","iGuardLevel":-1,"iNobleLevel":5,"sLogo":"http:\/\/downhdlogo.yy.com\/hdlogo\/144144\/3\/45\/81\/1453814503\/u1453814503kx70c9C.png","iUserLevel":9},{"lUid":2280652606,"sNickName":"\u5b9d\u8d1d\u9edb\u59ae","iScore":"966600","iGuardLevel":-1,"iNobleLevel":6,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1004\/fe\/23b188e487e66c052f7b315825e143_180_135.jpg?1544577796","iUserLevel":14},{"lUid":"1790788337","sNickName":"\u773a\u671b\u4f2f\u7eb3\u4e4c","iScore":"758000","iGuardLevel":-1,"iNobleLevel":2,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1089\/08\/9bc8107fc151a2c6707629f2a7cb99_180_135.jpg?1540608386","iUserLevel":18},{"lUid":2241310128,"sNickName":"\ud83d\udc30\u82b1\u706b","iScore":"507500","iGuardLevel":-1,"iNobleLevel":2,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1017\/32\/d5879d93030896ec4baa27b3c14bc0_180_135.jpg?1544348968","iUserLevel":10},{"lUid":2305884846,"sNickName":"\u738b\u603b\u7684\u9648\u79d8\u4e66\u3001007","iScore":"500000","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1092\/fd\/cb2a4f373166c4fd3de28c7eaad674_180_135.jpg?1533813560","iUserLevel":10},{"lUid":"1795633308","sNickName":"\u56e9 \u98db \u63da","iScore":"399000","iGuardLevel":-1,"iNobleLevel":3,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1089\/d2\/86041062447a1a51e683b631e2580a_180_135.jpg","iUserLevel":4},{"lUid":"86057704","sNickName":"\u767d\u8587\u8587\u5bb6\u7684\u5b88\u62a4\u4e37\u591c\u6b87","iScore":"366000","iGuardLevel":-1,"iNobleLevel":5,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1090\/57\/6bbbe63af9dc1b628efde103a229d0_180_135.jpg?1532207904","iUserLevel":15},{"lUid":2297263912,"sNickName":"leftsun007","iScore":"351300","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1081\/71\/ccee1b6d6f5b946c896f7c0c05da84_180_135.jpg","iUserLevel":9},{"lUid":2230342791,"sNickName":"\u5f20\u81ea\u529b","iScore":"349600","iGuardLevel":-1,"iNobleLevel":4,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1044\/5d\/1b24844d57f3db1495f61859fdad84_180_135.jpg?1543707871","iUserLevel":18},{"lUid":"1800420257","sNickName":"\u661f\u7a7a\u4e2d\u6700\u4f4e\u8c03\u7684\u661f\u2764007","iScore":"251900","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1090\/2d\/bcfb6cc4a6099032bd6ad37f790c25_180_135.jpg?1511116881","iUserLevel":20},{"lUid":"1215610876","sNickName":"\u5983\u5b50\u5927\u5a76","iScore":"201800","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1018\/45\/c3f725c86d8758f63d79c74f18dd3f_180_135.jpg?1522106039","iUserLevel":9},{"lUid":"121366295","sNickName":"\u4e71\u4e71","iScore":"151400","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"","iUserLevel":3},{"lUid":2336580614,"sNickName":"\u8840\u796d\u591c","iScore":"146800","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1055\/44\/8f8d3732c8c19fa79bf7f429cfcf53_180_135.jpg?1543069259","iUserLevel":8},{"lUid":"1826096278","sNickName":"\u2728\u6d45\u7b11\u306e\u6bcd\u8001\u864e\u2728\ud83c\udf19","iScore":"132000","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1093\/36\/2f43557be3c48cdb2ee0d476902419_180_135.jpg?1526427863","iUserLevel":15},{"lUid":"84584019","sNickName":"\u7238\u7238\u6c60","iScore":"131600","iGuardLevel":-1,"iNobleLevel":2,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1090\/f8\/e904a947a3b17b572f3a3b69ccb36c_180_135.jpg?1537696200","iUserLevel":13},{"lUid":2183210913,"sNickName":"\u72fc\u4eba\u6740-\u53ef\u8fbe\u9e2d","iScore":"130000","iGuardLevel":-1,"iNobleLevel":2,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1016\/64\/d1cb4dda6bb2f37623d323b46ac617_180_135.jpg","iUserLevel":7},{"lUid":"153106347","sNickName":"\u610f\u8bc6\u672c\u6e90","iScore":"114500","iGuardLevel":-1,"iNobleLevel":4,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1049\/6e\/76f7944c0cf45b68f49161dfa7c542_180_135.jpg?1513010645","iUserLevel":21},{"lUid":"172548778","sNickName":"\u8bf7\u53eb\u6211\u5f6c\u5c11","iScore":"109000","iGuardLevel":1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1094\/af\/c42a64d9feedc427f8f3d61a7a33cb_180_135.jpg?1542377256","iUserLevel":9},{"lUid":"1828822412","sNickName":"\u5927\u80d6\u5189","iScore":"103100","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1090\/b1\/ad8b659e07ba96f5e3ba915769be24_180_135.jpg?1544552145","iUserLevel":10},{"lUid":"395304918","sNickName":"\u751c\u871c\u8f7b\u8bed","iScore":"101000","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1047\/cb\/e1c939b87bc28c5f20d1bed6a33413_180_135.jpg?1527497931","iUserLevel":20},{"lUid":2246306958,"sNickName":"\u987e\u4e5d","iScore":"100800","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"","iUserLevel":4},{"lUid":"1857552782","sNickName":"\u8001\u4e01","iScore":"99000","iGuardLevel":-1,"iNobleLevel":2,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1083\/55\/4c593ea68aed6a7b936f0957520d91_180_135.jpg?1542981146","iUserLevel":14},{"lUid":"1760738204","sNickName":"Amber","iScore":"99000","iGuardLevel":-1,"iNobleLevel":4,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1028\/27\/f8164b44402d0a75bd80437504ddfc_180_135.jpg?1500786174","iUserLevel":17},{"lUid":"675688343","sNickName":"\u5c0f\u5e03\u5494\u4e36007","iScore":"98400","iGuardLevel":12,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1057\/9b\/7b301f64e2fdb030b4139b4227d8f3_180_135.jpg?1531998024","iUserLevel":3},{"lUid":"855143","sNickName":"\u65e5\u6708\u5148\u751f","iScore":"98100","iGuardLevel":1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1017\/ce\/3950960501cc2e69db9b1869023e33_180_135.jpg?1544452566","iUserLevel":3},{"lUid":2328728555,"sNickName":"\u7231\u9a97\u8c46\u7684\u4e3b\u64ad","iScore":"94000","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1000\/52\/fc81ca0c678c704a0b3620d02172c5_180_135.jpg?1544028770","iUserLevel":8},{"lUid":2221997369,"sNickName":"\u597d@","iScore":"93800","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1097\/16\/12ed5f2b471bbdd5120b525b252cb3_180_135.jpg","iUserLevel":4},{"lUid":"1839275582","sNickName":"\u7cef\u7c73\u2714\ufe0f","iScore":"88000","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1048\/81\/2890041471644ef3e1266e40e7c276_180_135.jpg?1534107293","iUserLevel":11},{"lUid":1199511957173,"sNickName":"\u9592\u96f2\u5343\u91cc","iScore":"87700","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1021\/3b\/8015dcf44e849c0e6d196a8f2aa3ed_180_135.jpg?1543316216","iUserLevel":7},{"lUid":"1852183116","sNickName":"\u5361\u592b\u5361","iScore":"79200","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1078\/5d\/0808e4c731fe759dae5f97827291d6_180_135.jpg","iUserLevel":12},{"lUid":"1512337925","sNickName":"\u73a9\u73a9\u800c\u5df2","iScore":"66200","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1069\/80\/2010dd3003c63802269e482ba10992_180_135.jpg?1544942212","iUserLevel":10},{"lUid":"1621306730","sNickName":"\u6df9\u4e0d\u6b7b\u7684\u5c0f\u5f3a\uff5e","iScore":"66000","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1072\/60\/e0d2c51767c77597f3e7c03f8f07f9_180_135.jpg","iUserLevel":4},{"lUid":2354144385,"sNickName":"\u78a7\u7389","iScore":"61000","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1066\/15\/9cda6e746ed18b3bed2fd7efe04436_180_135.jpg?1544900349","iUserLevel":2},{"lUid":2288097566,"sNickName":"\u5154\u55b5\u9171\ud83d\udcad","iScore":"60400","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1072\/a6\/fb42432bfa2f987ecc08b093241594_180_135.jpg?1530782811","iUserLevel":2},{"lUid":"1753870610","sNickName":"\u5c0f\u9c7c\u513f\u4e0e\u82b1\u65e0\u7f3a","iScore":"56800","iGuardLevel":-1,"iNobleLevel":2,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1058\/5c\/6530209b89c71c5903dfa0785540c2_180_135.jpg?1542547125","iUserLevel":12},{"lUid":"71220572","sNickName":"\u6c5f\u5c0f\u767d","iScore":"55000","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1081\/39\/c67f9285142f09391e6b2097efcd11_180_135.jpg?1528555759","iUserLevel":11},{"lUid":1199512289995,"sNickName":"\u6797\u6069\u554a\u6797\u6069","iScore":"54300","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1000\/1e\/e1bcc56503f31b24dc6757524a9597_180_135.jpg?1543604136","iUserLevel":6},{"lUid":2331740740,"sNickName":"\u534e\u54e5","iScore":"51200","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1002\/6c\/d165e988afb3eff8c46721c40b3c92_180_135.jpg?1539953018","iUserLevel":7},{"lUid":"1830454546","sNickName":"Edin","iScore":"48300","iGuardLevel":-1,"iNobleLevel":2,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1016\/4a\/dc2a080994b08aa0443a746382302b_180_135.jpg","iUserLevel":3},{"lUid":"513835362","sNickName":"\u9759\u4f24","iScore":"46200","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1071\/b8\/1ff47dc9a0832a649825d0ecfeac40_180_135.jpg?1502805651","iUserLevel":5},{"lUid":1199512603198,"sNickName":"\u60c5\u4e4b\u6240\u6df1\u662f\u4f60","iScore":"44600","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1013\/b0\/f08f91cc7e20956a5ac9be6e418356_180_135.jpg?1544882372","iUserLevel":2},{"lUid":2320304646,"sNickName":"\u73b2\u5150","iScore":"42600","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1014\/9a\/7e2ca2c53876cca606505820a1c086_180_135.jpg?1544886648","iUserLevel":5},{"lUid":2339065281,"sNickName":"\u6211\u662f\u4e00\u9897\u5c0fJY","iScore":"37700","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1024\/0d\/2b06c604cd0677d69edbda456c8d1c_180_135.jpg","iUserLevel":4},{"lUid":2186360705,"sNickName":"\u4ece\u5c0f\u5c31\u5f88\u62a0\u95e8","iScore":"37600","iGuardLevel":-1,"iNobleLevel":4,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1022\/71\/1170af463c3d1803aa0c812f87272d_180_135.jpg?1544197148","iUserLevel":14},{"lUid":2252665268,"sNickName":"\u4f01\u9e45\u5d3d","iScore":"35000","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1078\/40\/8ed36610a9012151ebad3693ced28a_180_135.jpg","iUserLevel":2},{"lUid":"127928908","sNickName":"\u5962\u671b","iScore":"34400","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"","iUserLevel":5},{"lUid":"433955821","sNickName":"\u51b7\u6cea\u3001007","iScore":"33600","iGuardLevel":-1,"iNobleLevel":1,"sLogo":"https:\/\/huyaimg.msstatic.com\/avatar\/1091\/df\/430d3a637343414ed2622b57ccd6f0_180_135.jpg?1544791105","iUserLevel":14},{"lUid":2195338532,"sNickName":"\u5c0f\u7490\u5b50\ud83c\udf40","iScore":"31400","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"http:\/\/huyaimg.msstatic.com\/avatar\/1035\/91\/8acc37910a839430985f627e248f19_180_135.jpg?1514111054","iUserLevel":5},{"lUid":"413081565","sNickName":"11","iScore":"30400","iGuardLevel":-1,"iNobleLevel":-1,"sLogo":"","iUserLevel":2}],"iStart":1,"iCount":50,"iTotal":333},"ts":1544951416
}
```

http://www.huya.com/cache1min.php?m=VipBarList&tid=90126358&sid=90126358

参数：

```json
{
  "m": "VipBarList",
  "tid": 90126358,
  "sid": 90126358
}
```

响应：（获取贵宾席列表）

```json
{
    status: 1000
    result: {…}
    total: 426
    sBadgeName: 申屠
    items	[…]
        0	{…}
            iPotentialTypes	1
            iSuperPupleLevel	0
            iTypes	13
            lUid	1800751396
            sNickName	酒保舌头伸
            iUserLevel	18
            tFansInfo	{…}
                lUid	1800751396
                lBadgeId	1774581086
                iBadgeLevel	20
                iScore	85788
                iVFlag	1
                iBadgeType	0
            tGuardInfo	{…}
                lUid	0
                lPid	0
                iGuardLevel	0
                lEndTime	0
            tNobleInfo	{…}
                lUid	1800751396
                lPid	951238401
                lValidDate	1553011200
                sNobleName	公爵
                iNobleLevel	4
                iNoblePet	4
                iNobleStatus	2
                iNobleType	0
            tGuildMemInfo	{…}
                iGuildVip	1
        1	{…}
            iPotentialTypes	1
            iSuperPupleLevel	0
            iTypes	9
            lUid	1774581086
            sNickName	大申屠007
            iUserLevel	9
            tFansInfo	{…}
            tGuardInfo	{…}
            tNobleInfo	{…}
            tGuildMemInfo	{…}
        2	{…}
            iPotentialTypes	1
            iSuperPupleLevel	0
            iTypes	5
            lUid	65731892
            sNickName	心、一直在想念
            iUserLevel	16
            tFansInfo	{…}
            tGuardInfo	{…}
            tNobleInfo	{…}
            tGuildMemInfo	{…}
        3	{…}
            iPotentialTypes	1
            iSuperPupleLevel	0
            iTypes	1
            lUid	12379429
            sNickName	药药的大猪蹄子
            iUserLevel	11
            tFansInfo	{…}
            tGuardInfo	{…}
            tNobleInfo	{…}
            tGuildMemInfo	{…}
        4	{…}
        5	{…}
        6	{…}
        7	{…}
        8	{…}
        9	{…}
        10	{…}
        11	{…}
        12	{…}
        13	{…}
        14	{…}
        15	{…}
        16	{…}
        17	{…}
        18	{…}
        19	{…}
        20	{…}
        21	{…}
        22	{…}
        23	{…}
        24	{…}
        25	{…}
        26	{…}
        27	{…}
        28	{…}
        29	{…}
        30	{…}
        31	{…}
        32	{…}
        33	{…}
        34	{…}
        35	{…}
        36	{…}
        37	{…}
        38	{…}
        39	{…}
        40	{…}
        41	{…}
        42	{…}
        43	{…}
        44	{…}
        45	{…}
        46	{…}
        47	{…}
        48	{…}
        49	{…}
        50	{…}
        51	{…}
        52	{…}
        53	{…}
        54	{…}
        55	{…}
        56	{…}
        57	{…}
        58	{…}
        59	{…}
        60	{…}
        61	{…}
        62	{…}
        63	{…}
        64	{…}
        65	{…}
        66	{…}
        67	{…}
        68	{…}
        69	{…}
        70	{…}
        71	{…}
        72	{…}
        73	{…}
        74	{…}
        75	{…}
        76	{…}
        77	{…}
        78	{…}
        79	{…}
        80	{…}
        81	{…}
        82	{…}
        83	{…}
        84	{…}
        85	{…}
        86	{…}
        87	{…}
        88	{…}
        89	{…}
        90	{…}
        91	{…}
        92	{…}
        93	{…}
        94	{…}
        95	{…}
        96	{…}
        97	{…}
        98	{…}
        99	{…}
    }
    sVLogo: ""
}
```

http://www.huya.com//cache10min.php?m=Search&do=getHotword&v=2&callback=huyaNavPlaceholder

参数：

```json
{
  "m": "Search",
  "do": "getHotword",
  "v": 2,
  "callback": "huyaNavPlaceholder",
}
```

响应：（热搜关键字）

```json
{
    {"associate_id":"990112","hot_word":"\u5947\u602a\u541b","word_type":1},
    {"associate_id":"935168","hot_word":"\u4e0d\u6c42\u4eba","word_type":1},
    {"associate_id":"7911","hot_word":"\u97e6\u795e","word_type":1},
    {"associate_id":"660513","hot_word":"\u96be\u8a00","word_type":1},
    {"associate_id":"634853","hot_word":"AG\u5305\u5b50","word_type":1},
    {"associate_id":"520521","hot_word":"\u8f69\u5b50\u5de82\u5154","word_type":1},
    {"associate_id":"199300","hot_word":"\u9648\u5b50\u8c6a","word_type":1},
    {"associate_id":"3203","hot_word":"\u523a\u6fc0\u6218\u573a","word_type":2},
    {"associate_id":"121715","hot_word":"\u5fc3\u6001","word_type":1},
    {"associate_id":"2135","hot_word":"\u4e00\u8d77\u770b","word_type":2}
}
```

http://www.huya.com//cache10min.php?m=Game&do=ajaxNavGame&callback=huyaNavCategory

参数：

```json
{
  "m": "Game",
  "do": "ajaxNavGame",
  "callback": "huyaNavCategory"
}
```

响应：（导航分类列表）

```json
huyaNavCategory(
    {
        "status":1000,"result":{"hot":[{"host":"lol","name":"\u82f1\u96c4\u8054\u76df","isHot":0},{"host":"2793","name":"\u7edd\u5730\u6c42\u751f","isHot":0},{"host":"wzry","name":"\u738b\u8005\u8363\u8000","isHot":0},{"host":"xingxiu","name":"\u661f\u79c0","isHot":0},{"host":"seeTogether","name":"\u4e00\u8d77\u770b","isHot":0},{"host":"3203","name":"\u523a\u6fc0\u6218\u573a","isHot":0},{"host":"huwai","name":"\u6237\u5916","isHot":0},{"host":100032,"name":"\u4e3b\u673a\u6e38\u620f","isHot":0},{"host":"cf","name":"\u7a7f\u8d8a\u706b\u7ebf","isHot":0}],"user":[{"host":"dota2","name":"DOTA2","isHot":0},{"host":"3307","name":"\u5168\u519b\u51fa\u51fb","isHot":0},{"host":"wd","name":"\u95ee\u9053","isHot":0},{"host":"3793","name":"\u97f3\u4e50","isHot":0},{"host":"3189","name":"\u65e0\u9650\u6cd5\u5219","isHot":0},{"host":"wow","name":"\u9b54\u517d\u4e16\u754c","isHot":0},{"host":"overwatch","name":"\u5b88\u671b\u5148\u950b","isHot":0},{"host":"2952","name":"\u9006\u6c34\u5bd2","isHot":0},{"host":"4423","name":"Artifact","isHot":0}]}
    }
)
```

http://www.huya.com//cache10min.php?m=PResource&do=ajaxGetPResource&type=4&area=24&num=1&callback=huyaNavPResource

参数：

```json
{
  "m: "PResource",
  "do": "ajaxGetPResource",
  "type": 4,
  "area": 24,
  "num": 1,
  "callback": "huyaNavPResource"
}
```

响应：

```json
huyaNavPResource(
  {
    "status":1000, "data":[], "msg":""
  }
)
```


http://www.huya.com/cache5min.php?m=WhiteListApi&do=getWhiteList&type=webVersion

参数：

```json
{
  "m": "WhiteListApi",
  "do": "getWhiteList",
  "type": "webVersion"
}
```

响应：（虎牙网页版 Web 版本号）

```json
{
  "status":200, "data":"201812", "msg":""
}
```

https://liveapi.huya.com/moment/getCommentList?callback=jQuery1111027220565463250956_1544966616807&uid=&parentId=6634417355361011581&momId=6634417355361011581&lastComId=0&isGetHotComment=1&_=1544966616810

参数：



响应：（返回公屏聊天内容）

```json
{
    "status":200,"msg":"","data":{"comment":[{"comId":"6635579432326306420","parentId":"6634417355361011581","momId":"6634417355361011581","uid":2331819840,"nickName":"以梦为马只愿不负匆匆岁月","iconUrl":"https://huyaimg.msstatic.com/avatar/1054/f2/0543af1c998ce5ff72f41cb0bd4966_180_135.jpg?1538926953","content":"。。。","cTime":1544966230,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635533085355058016","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1813606797,"nickName":"LBJ！！！","iconUrl":"http://huyaimg.msstatic.com/avatar/1029/7a/c8101822cc8f3cd575d15ccc37ef65_180_135.jpg","content":"个、rf'hj】y/.ol","cTime":1544955439,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635527991466288280","parentId":"6634417355361011581","momId":"6634417355361011581","uid":2286897936,"nickName":"虎牙搜索虎娃","iconUrl":"https://huyaimg.msstatic.com/avatar/1042/1f/4198e3da30390be510ed9e87109f28_180_135.jpg?1531878742","content":"神经病","cTime":1544954253,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635500851604083110","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1378844169,"nickName":"故得美人兮","iconUrl":"https://huyaimg.msstatic.com/avatar/1087/6f/5e723cc467edcf5091807091d62607_180_135.jpg?1527206085","content":"顶我/{tx/{tx","cTime":1544947934,"favorCount":1,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635455561118291224","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1631684926,"nickName":"疯狂的史蒂夫","iconUrl":"https://huyaimg.msstatic.com/avatar/1093/69/6c479140e899fdf914e90cf0ed40b5_180_135.jpg?1530197945","content":"如果没人说的话，我是真以为是个女的","cTime":1544937389,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635454761833388152","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1701024669,"nickName":"波波90098","iconUrl":"https://huyaimg.msstatic.com/avatar/1038/80/3c9f6093985d0250dc826cc4ffc65d_180_135.jpg?1542519709","content":"/{sh/{sh/{sh/{sh/{sh","cTime":1544937203,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635336461645947655","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1700050879,"nickName":"小骚巷","iconUrl":"https://huyaimg.msstatic.com/avatar/1031/9c/9f3f3959d8838e46422bd4f644bf82_180_135.jpg?1537755773","content":"认识一下吧，我叫小骚巷点个订阅❤ 陪伴从这一刻开始 正在直播💕  ","cTime":1544909659,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635279261247710281","parentId":"6634417355361011581","momId":"6634417355361011581","uid":2247827938,"nickName":"路","iconUrl":"https://huyaimg.msstatic.com/avatar/1066/a0/96902b1500a9358a0d47bbd1f73601_180_135.jpg?1541867399","content":"年妹妹这里有托，小心","cTime":1544896341,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635243780476505002","parentId":"6634417355361011581","momId":"6634417355361011581","uid":2324902644,"nickName":"轩老大","iconUrl":"https://huyaimg.msstatic.com/avatar/1087/04/23a3f698c4164de1f2eef358ee9182_180_135.jpg?1544676403","content":"原谅我打个广告。。。我是一名刺激战场的新主播，刚刚直播人气不高。有没有一起玩的，希望大家多多支持。","cTime":1544888080,"favorCount":3,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6635226501806872797","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1695447522,"nickName":"网友","iconUrl":"https://huyaimg.msstatic.com/avatar/1071/f0/728ef1ff1d8b4f9977d1d205668f0a_180_135.jpg?1543226360","content":"BGM是QQ飞车的BGM好像，歌名忘了","cTime":1544884057,"favorCount":4,"replyCount":1,"comment":[{"comId":"6635478655169955444","parentId":"6635226501806872797","momId":"6634417355361011581","uid":1650166846,"nickName":"lyh","iconUrl":"http://huyaimg.msstatic.com/avatar/1049/46/f2f9d5165d4c0f3209ba40faa1c88b_180_135.jpg?1478787605","content":"over my head","cTime":1544942766,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0}],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0}],"lastComId":"6635226501806872797","hotComment":[{"comId":"6634430270350344142","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1802914684,"nickName":"浅笑安然","iconUrl":"https://huyaimg.msstatic.com/avatar/1052/1f/a635c86befb7f70e8435f2899342b9_180_135.jpg?1541415823","content":"我是被封面的陈子豪女装的骗进来的","cTime":1544698670,"favorCount":163,"replyCount":7,"comment":[{"comId":"6635553903075555815","parentId":"6634430270350344142","momId":"6634417355361011581","uid":1824645249,"nickName":"勇士酷旋风","iconUrl":"http://huyaimg.msstatic.com/avatar/1069/ff/16f89f28c513ee508476063835ddd9_180_135.jpg","content":"我也是","cTime":1544960286,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"6634430270350344142","replyToUid":"1802914684","replyToNickName":"浅笑安然","status":0,"opt":0},{"comId":"6635323894534722917","parentId":"6634430270350344142","momId":"6634417355361011581","uid":1792299312,"nickName":"云音阁丨子夜","iconUrl":"https://huyaimg.msstatic.com/avatar/1000/76/dd1a0db1c18a194fdc199d96e70596_180_135.jpg?1530560692","content":"+1","cTime":1544906733,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6634736733571401987","parentId":"6634430270350344142","momId":"6634417355361011581","uid":1598738890,"nickName":"浪子","iconUrl":"http://huyaimg.msstatic.com/avatar/1058/7d/aa21a77a6ead3ee175995762e1b831_180_135.jpg","content":"借楼bgm是什么？","cTime":1544770024,"favorCount":4,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0}],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6634438778691057038","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1626130383,"nickName":"藹鑈","iconUrl":"https://huyaimg.msstatic.com/avatar/1018/14/9ddc620d47d83d2b143c9a19106c50_180_135.jpg?1540472696","content":"被封面骗进来的举一下手","cTime":1544700651,"favorCount":115,"replyCount":5,"comment":[{"comId":"6635051632104603660","parentId":"6634438778691057038","momId":"6634417355361011581","uid":2235317904,"nickName":"自在h哲理","iconUrl":"http://huyaimg.msstatic.com/avatar/1042/0f/630c696d6f63655c7e80a888103cda_180_135.jpg?1522468659","content":"是的","cTime":1544843342,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6634947169882441773","parentId":"6634438778691057038","momId":"6634417355361011581","uid":2189366304,"nickName":"幽天月","iconUrl":"http://huyaimg.msstatic.com/avatar/1056/5d/3305c200acf3da174a33269105da38_180_135.jpg","content":"+1","cTime":1544819020,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6634818432484807753","parentId":"6634438778691057038","momId":"6634417355361011581","uid":2222224834,"nickName":"  爆头-猪猪","iconUrl":"http://downhdlogo.yy.com/hdlogo/144144/34/48/22/2222224834/u2222224834l5ecCq1.jpg","content":"对\\(^o^)/","cTime":1544789046,"favorCount":0,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0}],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0},{"comId":"6634444018520913441","parentId":"6634417355361011581","momId":"6634417355361011581","uid":1860709374,"nickName":"黄金星","iconUrl":"https://huyaimg.msstatic.com/avatar/1076/f3/9e154c5668c9cd72c83b77533f8400_180_135.jpg?1528074506","content":"能不能长一点的，两秒一剪切，太快了没有看点","cTime":1544701871,"favorCount":31,"replyCount":1,"comment":[{"comId":"6634444426543018128","parentId":"6634444018520913441","momId":"6634417355361011581","uid":1860709374,"nickName":"黄金星","iconUrl":"https://huyaimg.msstatic.com/avatar/1076/f3/9e154c5668c9cd72c83b77533f8400_180_135.jpg?1528074506","content":"说错了是三秒/{xk","cTime":1544701966,"favorCount":1,"replyCount":0,"comment":[],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0}],"replyToComId":"0","replyToUid":"0","replyToNickName":"","status":0,"opt":0}]}
}
```
