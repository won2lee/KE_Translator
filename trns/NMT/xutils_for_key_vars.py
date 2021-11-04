def make_key_vars():
    # 유니코드 한글 시작 : 44032, 끝 : 55199
    BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

    # 초성 리스트. 00 ~ 18
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 
                    'ㅍ', 'ㅎ']

    # 중성 리스트. 00 ~ 20
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 
                     'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
                     'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    ch_len = max([len(s) for s in [CHOSUNG_LIST,JUNGSUNG_LIST, JONGSUNG_LIST]])


    kor_alpha = CHOSUNG_LIST + ['#']*(ch_len - len(CHOSUNG_LIST)) + JUNGSUNG_LIST + ['#']*(ch_len - len(JUNGSUNG_LIST))+JONGSUNG_LIST

    kor_alpha = kor_alpha+[s for s in '0123456789-_']

    len_alpha =96+BASE_CODE
    ch_start = chr(len_alpha-1)
    ch_end = chr(len_alpha-1)
    #ch_end = chr(len_alpha-1)

    s_to_double, d_to_single, ch_to_num, num_to_ch = double_vowel_lookup2(JUNGSUNG_LIST)

    double_vowel =[]
    for c in 'ㅕㅘㅝㅟㅐ':
        double_vowel.append(ch_to_num[c])

    k_alpha_to_num =[{},{}]
    for i, s in enumerate(kor_alpha[28:]):
        k_alpha_to_num[1][s] = i+28 
    for i, s in enumerate(kor_alpha[:56]):
        k_alpha_to_num[0][s] = i 
    for i in range(2):
        k_alpha_to_num[i].pop(' ',1)
        k_alpha_to_num[i].pop('#',1)

    key_vars = {'BASE_CODE':BASE_CODE,'CHOSUNG':CHOSUNG, 'JUNGSUNG':JUNGSUNG,'ch_len':ch_len,
                'kor_alpha':kor_alpha, 'len_alpha':len_alpha, 'ch_start':ch_start,
                'CHOSUNG_LIST':CHOSUNG_LIST,'JUNGSUNG_LIST':JUNGSUNG_LIST,'JONGSUNG_LIST':JONGSUNG_LIST,
                'double_vowel':double_vowel, 's_to_double':s_to_double, 'd_to_single':d_to_single,
                'k_alpha_to_num':k_alpha_to_num }
    return key_vars

def double_vowel_lookup2(JUNGSUNG_LIST):
    vowels = JUNGSUNG_LIST
    s = [c for c in'ㅕㅘㅝㅟㅐㅑㅒㅖㅙㅚㅛㅞㅠㅢ']
    d = 'ㅣㅓ ㅗㅏ ㅜㅓ ㅜㅣ ㅏㅣ ㅣㅏ ㅣㅐ ㅣㅔ ㅗㅐ ㅗㅣ ㅣㅗ ㅜㅔ ㅣㅜ ㅡㅣ'.split(' ')
    s_to_d = {}

    for k,v in list(zip(s,d)): 
        s_to_d[k] = v

    ch_to_num = {}
    for i, c in enumerate(vowels):
        ch_to_num[c] = i
    #print(ch_to_num)
    
    num_to_ch = {}
    for k, v in ch_to_num.items():
        num_to_ch[v] = k

    s_to_double = {}
    d_to_singleN = {}
    for k,v in s_to_d.items():
        s_to_double[ch_to_num[k]] = tuple([ch_to_num[ch] for ch in v])

    for k,v in s_to_double.items():
        d_to_singleN[v] = k

    return s_to_double, d_to_singleN, ch_to_num, num_to_ch
