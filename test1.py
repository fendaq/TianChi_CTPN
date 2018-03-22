import os,random,re,sys
from PIL import Image,ImageDraw,ImageFont
from collections import Counter

# 背景图地址
image_path = 'images_base/bkgd_5_2.png'
# 字体地址
list_fonts = ['fonts/tecnico_bold.ttf',
              'fonts/tecnico_bolditalic.ttf',
              'fonts/tecnico_regular.ttf',
              'fonts/AnisaSans.ttf',
              'fonts/SEA_GARDENS.ttf']
# 这是啥
words_file = 'dictionary.txt'
#
dir_data_generated = 'data_generated'
#
dir_images_gen = dir_data_generated + '/images'
dir_contents_gen = dir_data_generated + '/contents'
#
list_sizes = [26, 28, 30, 31, 32, 33, 34, 36, 40, 44]
#新建文件夹
if not os.path.exists(dir_data_generated): os.mkdir(dir_data_generated)
if not os.path.exists(dir_images_gen): os.mkdir(dir_images_gen)
if not os.path.exists(dir_contents_gen): os.mkdir(dir_contents_gen)

def extractWords(text):
    return re.findall(r'\w+', text.lower())

WORDS = Counter(extractWords(open(words_file, encoding="utf-8").read()))
# print(WORDS)

def generateRandomDigital():
    #
    a = random.randint(0, 10)
    b = random.randint(0, 999)
    c = random.randint(0, 999)
    d = random.randint(0, 99)
    #
    s = '0'
    #
    if random.random() < 0.5:
        d = 0
    #
    if a > 0 and random.random() < 0.1:
        a = '%d' % a
        b = '%03d' % b
        c = '%03d' % c
        d = '%02d' % d
        #
        s = a + ',' + b + ',' + c
        #
        if random.random() < 0.5: s += '.' + d
        #
    elif b > 0 and random.random() < 0.4:
        b = '%d' % b
        c = '%03d' % c
        d = '%02d' % d
        #
        s = b + ',' + c
        #
        if random.random() < 0.5: s += '.' + d
        #
    else:
        c = '%d' % c
        d = '%02d' % d
        #
        s = c
        #
        if random.random() < 0.5: s += '.' + d

    return s


def addDigitalWords(words_dict, num):
    #
    for i in range(int(num)):
        #
        words_dict[generateRandomDigital()] = 100
        #
    #
    return words_dict



WORDS = addDigitalWords(WORDS, len(WORDS) * 0.30)
list_words = list(WORDS.keys())
print(list_words)

def drawLineNoise(draw, width, height):
    #
    for i in range(100):
        if random.random() > 0.05:
            continue

        x0 = random.randint(0, width)
        x1 = random.randint(0, width)
        y0 = random.randint(0, height)
        y1 = random.randint(0, height)

        if random.random() < 0.40:
            y1 = y0
        line_width = random.randint(1, 4)

        draw.line([(x0, y0),(x1,y1)], fill=(0,0,0), width=line_width)



def drawPatchNoise(draw, x0, y0):
    #
    x1 = x0 + random.randint(20, 100)
    y1 = y0 + random.randint(20, 100)
    ym = (y0+y1)/2
    #
    draw.line([(x0,ym),(x1,ym)], fill = (0,0,0), width = y1-y0)
    #
    return [x1, y1]


def generateDataSample(draw, width, height):

    drawLineNoise(draw, width, height)
    width -= 36
    width -= 36

    list_bbox_text = []

    x0 = 36 + random.randint(0, 100)
    y0 = 36 + random.randint(0, 100)

    x1 = x0
    y1 = y0

    max_th = 0

    while True:
        if random.random() < 0.05:
            ep = drawPatchNoise(draw, x0, y0)
            x0 = ep[0] + 50 + random.randint(0, 100)
            y1 = max(y1, ep[1])

        font_file = random.choice(list_fonts)
        text_size = random.choice(list_sizes)
        text_str = random.choice(list_words)

        font = ImageFont.truetype(font_file, text_size)
        # 文字大小
        tw, th = draw.textsize(text_str, font)

        if th > max_th: max_th = th
        #
        xs = x0  # - round(tw/50.0)
        ys = y0  # + round(th/20.0)
        ym = ys + th / 2.0  # round(th/2.0)
        #
        xe = x0 + tw
        ye = ys + th
        #
        x1 = xe
        y1 = max(y1, ye)

        if y1 > height: break
        if x1 > width:
            x0 = 36 + random.randint(0, 100)
            y0 = y1 + 1 + random.randint(0, 100)
            #
            x1 = x0
            y1 = y0
            #
            continue

        #清空背景
        draw.line([(x0,ym),(x0+tw,ym)], fill = (255,255,255), width = th)
        #画文字, 设置文字位置/内容/颜色/字体
        draw.text((x0,y0), text_str, (0, 0, 0), font=font)


        #画方框
        if random.random() < 0.20:
            line_width = 1 # round(text_size/10.0)
            draw.line([(xs,ys),(xs,ye),(xe,ye),(xe,ys),(xs,ys)],
                      width=line_width, fill=(0,0,0))
        #
        list_bbox_text.append([[xs, ys,
                                xs+tw, ys+th + round(th/10.0)], text_str])

        x0 = x1 + 50 + random.randint(0, 100)

    print('max text_height: %d' % max_th)
    #
    return list_bbox_text


count = 0

count += 1

for num in range(100):
    img_draw = Image.open(image_path)
    img_size = img_draw.size
    draw = ImageDraw.Draw(img_draw)
    list_bbox_text = generateDataSample(draw, img_size[0], img_size[1])
    del draw


    #另存图片
    imageTextFile = 'data_generated/images/hhh' + '_generated_' + str(num) + '.png'
    img_draw.save(imageTextFile)
    #
    #保存内容
    contentFile = 'data_generated/contents/hhh' + '_generated_' + str(num) + '.txt'
    with open(contentFile, 'w') as fp:
        for item in list_bbox_text:
            #
            fp.write('%d-%d-%d-%d|%s\n' %(item[0][0],item[0][1],item[0][2],item[0][3],item[1]))
            #


