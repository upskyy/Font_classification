import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

font_path = "C:\\Python Project\\font_project\\fonts_ttf"
fonts = os.listdir("C:\\Python Project\\font_project\\fonts_ttf")

# 한글 유니코드
co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
start = "AC00"
end = "D7A3"

co = co.split(" ")
Hangul_Syllables = [a+b+c+d 
                    for a in co 
                    for b in co 
                    for c in co 
                    for d in co]

Hangul_Syllables = np.array(Hangul_Syllables)

s = np.where(start == Hangul_Syllables)[0][0]
e = np.where(end == Hangul_Syllables)[0][0]

Hangul_Syllables = Hangul_Syllables[s : e + 1]


# 잘 불러왔나 확인하기 위해 이미지 출력

unicodeChars = chr(int(Hangul_Syllables[0], 16))  # 가

plt.figure(figsize=(15, 15))

for idx, ttf in enumerate(fonts):

    font = ImageFont.truetype(font = font_path +"\\"+ ttf, size = 100)

    x, y = font.getsize(unicodeChars)

    theImage = Image.new('RGB', (x + 3, y + 3), color='white')

    theDrawPad = ImageDraw.Draw(theImage)

    theDrawPad.text((0, 0), unicodeChars[0], font=font, fill='black')

    plt.subplot("24{}".format(str(idx + 1)))  #2*4로 subplot을 잡은 것중 1,2,3번째
    
    plt.title(str(ttf))  #이름
    
    plt.imshow(theImage)  
    
plt.show()  #출력


#이미지 파일로 저장하는 과정

for uni in tqdm(Hangul_Syllables):
    
    unicodeChars = chr(int(uni, 16))
    
    path = "C:\\Python Project\\font_project\\fonts_png" + unicodeChars  #기본 경로
    
    os.makedirs(path, exist_ok = True)
        
    for ttf in fonts:
        
        font = ImageFont.truetype(font = font_path +"\\"+ ttf, size = 100)
        
        x, y = font.getsize(unicodeChars)
        
        theImage = Image.new('RGB', (x + 3, y + 3), color='white') # 바탕 흰색
        # Image.new(mode, size, color) 
        theDrawPad = ImageDraw.Draw(theImage)
        
        theDrawPad.text((0.0, 0.0), unicodeChars[0], font=font, fill='black' )
        
        msg = path + "/" + ttf[:-4] + "_" + unicodeChars #각 이미지 이름
        
        theImage.save('{}.png'.format(msg))  # 이미지 저장

