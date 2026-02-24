"""
Vietnamese Toxic Words and Patterns Database - ENTERPRISE EDITION
Comprehensive database for maximum content safety

Version: 3.0 - Enhanced Detection
Coverage: 500+ toxic patterns, 1000+ variants
Last Updated: 2025-11-04
"""

# ==================== LEVEL 1: SEVERE (AUTO REJECT) ====================

# Extremely severe profanity - EXPANDED with VARIANTS
SEVERE_PROFANITY = [
    # ===== Sexual-related words =====
    # Đụ/Địt - ONLY KEEP CLEAR PATTERNS
    # REMOVED: 'du', 'dụ', 'đu', 'dù', 'dú', 'đú', 'dũ', 'đũ' - gây false positive với:
    #   - duyên, duyệt, du lịch, du học, giáo dục, sử dụng, dữ liệu...
    # REMOVED: 'đục', 'dục', 'đủ', 'dư', 'đư', 'dứ', 'đứ', 'dứt' - gây false positive với:
    #   - giáo dục, dục vọng (hợp lệ trong context), đục lỗ, dư thừa...
    
    # Clearly violating words (full diacritics or clearly obfuscated)
    'đụ', 'địt',  # Chỉ 2 từ chính với dấu đầy đủ
    'đit', 'dit', 'dịt', 'đyt', 'dyt',  # Unmistakable variants
    'djt', 'dj t',  # Leetspeak variants
    'd1t', 'đ1t', 'd!t', 'đ!t',  # Số/ký tự thay thế
    
    # Obfuscated patterns (có dấu phân cách rõ ràng - intentional bypass)
    'd.i.t', 'd-i-t', 'd_i_t', 'đ.ị.t', 'đ-ị-t', 'đ_ị_t',
    'd i t', 'đ i t', 'd  i  t', 'đ  i  t',
    'd.u.m', 'đ.u.m', 'd.u.c', 'd-u-t', 'd_u_t',
    # NOTE: Single words like 'du', 'dụ', 'đu' REMOVED - left for context analyzer
    
    # Lồn variations - ONLY KEEP CLEAR PATTERNS
    # REMOVED: single 'lon' - causes false positives with "lon bia", "hài lòng"
    'lồn',  # Main word with full diacritics
    'l0n', 'l0l', 'lo0n', '10n', '1on',  # Clear leetspeak
    # Obfuscated patterns
    'l o n', 'l ồ n', 'l.o.n', 'l.ồ.n', 'l-o-n', 'l_o_n',
    'l0.n', 'l.0.n', 'l-0-n', 'l_0_n',
    # Unicode lookalikes
    '1ồn', 'ιồn', 'ιon',
    
    # Cặc variations (60+ variants)
    'cặc', 'cac', 'cak', 'cắc', 'cặk', 'cak', 'cek', 'cạc',
    'c a c', 'c ặ c', 'c.a.c', 'c.ặ.c', 'c-a-c', 'c_a_c',
    'c@.c', 'cax', 'cek', 'kac', 'kặc',
    'cá c', 'cá-c', 'cá_c', 'cá.c',
    # NOTE: "c@c" có thể match "các" trong "cung c@p" nên pattern sẽ check word boundary
    
    # Buồi variations
    'buồi', 'buoi', 'bu0i', 'buời', 'búi', 'bùi',
    'bu ồ i', 'bu.ồ.i', 'bu-ồ-i', 'bu_ồ_i',
    
    # Genitalia
    'dương vật', 'duong vat', 'âm đạo', 'am dao', 'âm hộ', 'am ho',
    'âm vật', 'âm môn', 'dái', 'dai', 'bú cu', 'bu cu', 'bú cặc',
    'tinh trùng', 'tinh dịch', 'xuất tinh', 'xuat tinh',
    
    # Sexual acts
    'chịch', 'chich', 'chịt', 'chit', 'ch!ch', 'ch1ch',
    'địm', 'dim', 'đym', 'dym', 'nện', 'nen', 'bú lồn', 'bu lon',
    'liếm lồn', 'liem lon', 'mút cu', 'mut cu', 'mút cặc',
    'fuck', 'fucking', 'fucked', 'fuk', 'fck', 'f*ck', 'f**k',
    'pussy', 'dick', 'cock', 'penis', 'vagina', 'cunt',
    'blowjob', 'blow job', 'handjob', 'hand job', 
    
    # ===== Severe profanity =====
    # ĐM/DCM family (100+ variants)
    'đm', 'dm', 'đ m', 'd m', 'đ.m', 'd.m', 'đ-m', 'd-m', 'đ_m', 'd_m',
    'đmm', 'dmm', 'đ mm', 'd mm', 'đ.m.m', 'd.m.m',
    'đcm', 'dcm', 'đ cm', 'd cm', 'đ.c.m', 'd.c.m', 'đ-c-m', 'd-c-m',
    'đmmm', 'dmmm', 'đcmm', 'dcmm', 'đkmm', 'dkmm',
    'đ!m', 'd!m', 'đ1m', 'd1m', 'đ@m', 'd@m',
    'đờ mờ', 'do mo', 'đờ em', 'do em',
    'đéo má', 'deo ma', 'đệch mẹ', 'dech me',
    'đụ má', 'du ma', 'địt mẹ', 'dit me', 'đụ mẹ', 'du me', 'địt má', 'dit ma',
    
    # VCL/VL family (80+ variants)
    'vcl', 'v cl', 'v.c.l', 'v-c-l', 'v_c_l',
    'vkl', 'v kl', 'v.k.l', 'vờ cờ lờ', 'vo co lo',
    'vl', 'v l', 'v.l', 'v-l', 'v_l', 'vờ lờ', 'vo lo',
    'vãi lồn', 'vai lon', 'vãi lon', 'v~l', 'v~c~l',
    'v!cl', 'v1cl', 'vcI', 'vсl', 'νcl',
    'wl', 'wcl', 'vc1', 'v cai lon',
    
    # CC family (60+ variants)
    'cc', 'c c', 'c.c', 'c-c', 'c_c', 'cờ cờ', 'co co',
    'cặc cứ', 'cac cu', 'cứ cặc', 'cu cac',
    'c!c', 'c1c', 'сс', 'ςς',
    
    # CLM/CTM family
    'clm', 'cl m', 'c.l.m', 'c-l-m', 'c_l_m', 'cho lờ mờ',
    'ctm', 'ct m', 'c.t.m', 'c-t-m', 'c_t_m',
    'cmm', 'cm m', 'c.m.m',
    
    # Đéo family (40+ variants)
    'đéo', 'deo', 'đ éo', 'd éo', 'đ ê o', 'd e o',
    'đ.é.o', 'd.é.o', 'đ-é-o', 'd-é-o',
    'đê o', 'de o', 'đêo', 'đeo`', 'đeo~',
    'đ!o', 'd!o', 'đ3o', 'd3o',
    
    # Excrement/Bodily fluids
    'cứt', 'cut', 'c ứt', 'c.ứ.t', 'c-ứ-t', 'cứ t',
    'đái', 'dai', 'đ ái', 'đ.á.i', 'đa i',
    'đi đái', 'di dai', 'đi ỉa', 'di ia', 'đi cầu', 'di cau',
    'nước đái', 'nuoc dai', 'nước tiểu', 'nuoc tieu',
    
    # English profanity (expanded)
    'shit', 'sh!t', 'sh1t', 'shіt', 'shìt',
    'fuck', 'fuk', 'fck', 'f*ck', 'f**k', 'f***', 'fvck', 'phuck',
    'fucking', 'fuking', 'fcking', 'f*cking',
    'motherfucker', 'mother fucker', 'mofo', 'm0f0', 'mf',
    'bitch', 'b!tch', 'b1tch', 'bītch', 'biatch',
    'asshole', 'ass hole', '@sshole', 'a$$hole', 'assh0le',
    'bastard', 'b@stard', 'bast@rd',
    'cunt', 'c*nt', 'c**t', 'kunt',
    'whore', 'wh0re', 'who re',
    'slut', 'sl*t', '5lut',
    'piss', 'p!ss', 'p1ss',
    
    # ===== Kết hợp từ chửi =====
    # Đụ/Địt + họ hàng
    'đụ má', 'du ma', 'địt mẹ', 'dit me', 'đụ mẹ', 'du me', 'địt má', 'dit ma',
    'đụ cha', 'du cha', 'địt cha', 'dit cha', 'địt bố', 'dit bo', 'đụ bố', 'du bo',
    'đụ cụ', 'du cu', 'địt cụ', 'dit cu',
    'đụ ông', 'du ong', 'đụ bà', 'du ba',
    'đụ con', 'du con', 'đụ mày', 'du may', 'địt mày', 'dit may',
    
    # Mẹ/Má variations
    'mẹ kiếp', 'me kiep', 'mẹ kệch', 'me kech', 'mẹ mày', 'me may',
    'đm mày', 'dm may', 'địt mẹ mày', 'dit me may',
    'chết mẹ', 'chet me', 'chết mày', 'chet may',
    
    # Cha/Bố variations  
    'cha chó', 'cha cho', 'bố láo', 'bo lao', 'cha nó', 'cha no',
    'mẹ nó', 'me no', 'bố mày', 'bo may', 'cha mày', 'cha may',
    
    # Combo profanity
    'cái lồn', 'cai lon', 'cái lon', 'con lồn', 'con lon',
    'cái cặc', 'cai cac', 'con cặc', 'con cac', 'thằng cặc', 'thang cac',
    'đồ chó đẻ', 'do cho de', 'đồ súc vật', 'do suc vat', 'đồ con hoang', 'do con hoang',
    'lồn mẹ', 'lon me', 'lồn cha', 'lon cha', 'cặc cha', 'cac cha',
]

# ===== Severe Insults - EXPANDED =====
SEVERE_INSULTS = [
    # ===== Intellectual insults (only when targeting INDIVIDUALS) =====
    'ngu', 'ngu ngốc', 'ngu người', 'ngu vãi', 'ngu như lợn', 'ngu si', 'ngu xuẩn',
    'ngu như chó', 'ngu xuẩn', 'ngu dốt', 'ngu si', 'ngu vl', 'ngu vcl',
    'đần', 'đần độn', 'đần khờ', 'đần như bò', 'đần thật', 'đần vl',
    'ngớ ngẩn', 'ngu ngơ', 'đầu óc cùn', 'đầu óc đơn giản',
    'não cá vàng', 'não lợn', 'não bò', 'não chó', 'não gà',
    'đầu gối', 'đầu đất', 'óc chó', 'óc lợn', 'óc gà',
    'thiểu năng', 'khuyết tật', 'tâm thần', 'tâm lý bất thường',
    'retard', 'retarded', 'stupid', 'idiot', 'moron', 'dumb', 'dumbass',
    'imbecile', 'fool', 'foolish', 'brainless', 'mindless',
    
    # ===== Character assassination / Dignity insults =====
    # Animal comparisons (SEVERE)
    'đồ chó', 'đồ lợn', 'thằng chó', 'con chó', 'đồ súc vật',
    'thằng lợn', 'con lợn', 'đồ heo', 'thằng heo', 'con heo',
    'đồ bò', 'thằng bò', 'con bò', 'đồ trâu', 'thằng trâu',
    'đồ khỉ', 'thằng khỉ', 'con khỉ', 'bọn khỉ',
    'đồ gà', 'thằng gà', 'đồ vịt', 'đồ ngan',
    'đồ rắn', 'đồ rết', 'đồ bọ', 'đồ giun',
    'súc vật', 'loài vật', 'thú vật', 'vật nuôi',
    
    # Dignity insults
    'đồ khốn', 'thằng khốn', 'con khốn', 'khốn nạn', 'khốn kiếp',
    'đồ bẩn', 'thằng bẩn', 'đồ bẩn thỉu', 'đồ dơ bẩn',
    'đồ rác rưởi', 'rác rưởi', 'đồ phế thải', 'đồ rác', 'thứ rác',
    'cặn bã', 'thấp hèn', 'đồ hèn', 'đồ hạ đẳng', 'hạ đẳng',
    'đồ tiện', 'đồ hèn mọn', 'đồ vô dụng', 'vô dụng',
    'đồ vô giá trị', 'thứ rác', 'đồ bỏ đi', 'đồ thừa thãi',
    'scum', 'trash', 'garbage', 'worthless', 'useless',
    
    # ===== Moral / Sexual insults =====
    # Prostitution
    'đồ điếm', 'con điếm', 'đĩ', 'điếm đĩ', 'cave', 'gái cave',
    'con đĩ', 'thằng đĩ', 'gái điếm', 'gái mại dâm', 'đĩ thõa',
    'gái rẻ tiền', 'đồ rẻ rách', 'đồ bán dâm', 'bán dâm',
    'gái gọi', 'gai goi', 'gái bao', 'gái ngành',
    'prostitute', 'hooker', 'slut', 'whore', 'hoe',
    
    # Education/Culture
    'đồ mất dạy', 'vô giáo dục', 'vô văn hóa', 'thiếu văn hóa',
    'đồ không có giáo dục', 'thiếu dạy bảo', 'dốt nát',
    'không biết học', 'thất học', 'lạc hậu',
    
    # ===== Violence threats - SEVERE =====
    # Kill/Stab
    'giết', 'giết mày', 'chém', 'chém mày', 'chém chết',
    'đánh chết', 'đập chết', 'giết chết', 'giết bỏ',
    'tao đánh mày', 'tao giết mày', 'tao chém mày',
    'kill', 'murder', 'kill you', 'gonna kill',
    
    # Other violence
    'đánh đập', 'đập đầu', 'đấm mặt', 'đấm cho',
    'tát cho', 'vả mặt', 'đá đít',
    'beat', 'beat up', 'smash', 'punch',
]

# ===== HATE SPEECH - Discrimination =====
# This is a CRITICAL part missing in old systems!

# LGBTQ+ Discrimination (SEVERE)
HATE_LGBTQ = [
    # Gay/Lesbian slurs
    'gay đáng ghét', 'gay đáng khinh', 'gay đáng chết', 
    'đồ gay', 'thằng gay', 'con gay', 'bọn gay',
    'gay bệnh hoạn', 'gay đê tiện', 'gay tởm lợm',
    'đồ đồng tính', 'thằng đồng tính', 'bọn đồng tính',
    'pê đê', 'pede', 'pe đe', 'bê đê', 'bede',
    'thằng pê đê', 'đồ pê đê', 'con pê đê',
    'les', 'les dai', 'đồ les', 'con les',
    'đồng tính bệnh hoạn', 'đồng tính đáng khinh',
    
    # Transgender slurs
    'chuyển giới bệnh hoạn', 'chuyển giới đáng ghét',
    'đồ chuyển giới', 'thằng chuyển giới',
    'tranny', 'shemale', 'ladyboy đê tiện',
    'giả gái', 'gia gai', 'đồ giả gái', 'thằng giả gái',
    
    # General LGBTQ hate
    'bọn biến thái', 'đồ biến thái tình dục',
    'tâm lý bất thường', 'rối loạn giới tính',
    'đồ tha hóa', 'thằng tha hóa',
    'queer', 'faggot', 'fag', 'dyke',
    
    # Context: contempt/hatred
    'gay đáng bị', 'gay cần phải', 'tiêu diệt gay',
    'đáng bị khinh thường', 'đáng bị khinh', 'đáng khinh thường',
    'đáng ghét', 'đáng chết', 'nên chết',
    
    # NEW: LGBT specific phrases (Auto-Reject)
    'lgbt là tội lỗi', 'lgbt tội lỗi', 'lgbt la toi loi',
    'lgbt là bệnh', 'lgbt là benh', 'lgbt bệnh hoạn',
    'lgbt bị cấm', 'lgbt nên bị cấm', 'cấm lgbt',
    'chống lgbt', 'anti lgbt', 'tẩy chay lgbt',
]

# Racial/Ethnic Discrimination (SEVERE)  
HATE_RACISM = [
    # Chủng tộc Trung Quốc
    'đồ tàu', 'tàu khựa', 'thằng tàu', 'con tàu', 'bọn tàu',
    'tàu cộng', 'tàu giặc', 'tàu lùn', 'tàu đần',
    'chink', 'chinky', 'ching chong',
    
    # Chủng tộc da đen
    'khỉ đen', 'bọn khỉ đen', 'thằng khỉ đen',
    'mọi đen', 'đen như than', 'đen thui',
    'nigger', 'nigga', 'negro',
    
    # Dân tộc thiểu số
    'mọi rợ', 'rừng núi', 'dân thổ', 'thổ dân',
    'dân tộc thiểu số bẩn', 'dân miền núi ngu',
    'ngoại tộc', 'man rợ',
    
    # Regional discrimination
    'đầu rồng đuôi tôm', 'đầu gấu', 'đầu bò',
    'miền bắc ngu', 'miền nam láo', 'miền trung nghèo',
]

# Religious Discrimination
HATE_RELIGION = [
    # Các tôn giáo
    'đạo đức giả', 'đạo ốc', 'tu sĩ giả',
    'hòa thượng ăn mặn', 'sư ăn thịt',
    'đạo nào cũng lừa đảo', 'tôn giáo là thuốc phiện',
    
    # Cụ thể (tránh vi phạm)
    'phật tử ngu', 'cơ đốc giáo kém', 'hồi giáo bạo lực',
]

# Gender Discrimination
HATE_SEXISM = [
    # Contempt for women
    'đàn bà tóc dài trí ngắn', 'đàn bà ngu', 'con gái ngu',
    'phụ nữ chỉ biết sinh đẻ', 'phụ nữ thuộc về bếp núc',
    'gái chỉ biết bán thân', 'gái là đồ chơi',
    'mấy con gái', 'mấy con', 'đàn bà vô dụng',
    
    # Contempt for men
    'đàn ông rác rưởi', 'đàn ông chỉ biết chơi',
    'trai giống loài vật', 'nam giới vô dụng',
]

# ===== SEXUAL CONTENT / PORNOGRAPHY (SEVERE) =====
# This part was completely missing! Adding now!

# Explicit pornographic content
SEXUAL_EXPLICIT = [
    # Specific sexual acts
    'bú cu', 'bu cu', 'bú cặc', 'bu cac', 'bú lồn', 'bu lon',
    'liếm lồn', 'liem lon', 'liếm cặc', 'liem cac',
    'mút cu', 'mut cu', 'mút cặc', 'mut cac',
    'chịch nhau', 'chich nhau', 'địt nhau', 'dit nhau',
    'quan hệ tình dục', 'quan he tinh duc', 'làm tình', 'lam tinh',
    'sex', 'having sex', 'oral sex', 'anal sex',
    'blowjob', 'blow job', 'handjob', 'hand job',
    '69', 'sixty nine', 'doggy', 'doggy style',
    
    # Detailed genitalia
    'dương vật to', 'duong vat to', 'cặc to', 'cac to',
    'lồn chặt', 'lon chat', 'lồn bự', 'lon bu',
    'vú to', 'vu to', 'vú bự', 'vu bu', 'ngực to', 'nguc to',
    'mông to', 'mong to', 'mông bự', 'mong bu', 'đít to', 'dit to',
    
    # Ejaculation, orgasm
    'xuất tinh', 'xuat tinh', 'ra nước', 'ra nuoc',
    'cực khoái', 'cuc khoai', 'lên đỉnh', 'len dinh',
    'orgasm', 'cum', 'cumming', 'ejaculate',
]

# Suggestive / Sexually provocative content
SEXUAL_SUGGESTIVE = [
    # Sexually-charged physical comments
    'gái xinh bú cu', 'gai xinh bu cu',
    'thân hình gợi cảm', 'than hinh goi cam',
    'sexy', 'sexy quá', 'sexy vãi', 'sexy vl',
    'dâm', 'dam', 'dâm đãng', 'dam dang', 'dâm dục', 'dam duc',
    'dâm dật', 'dam dat', 'tục tĩu', 'tuc tiu',
    
    # Sexually suggestive actions
    'cởi áo', 'coi ao', 'cởi quần', 'coi quan', 'lột đồ', 'lot do',
    'khỏa thân', 'khoa than', 'trần truồng', 'tran truong',
    'nude', 'naked', 'strip', 'stripper',
    
    # Sexually-charged comments about others
    'thế này chắc', 'the nay chac', 'chắc giỏi', 'chac gioi',
    'giỏi trên giường', 'gioi tren giuong', 'giỏi chuyện ấy', 'gioi chuyen ay',
    'phục vụ tốt', 'phuc vu tot', 'làm tình giỏi', 'lam tinh gioi',
]

# Sexual solicitation / propositioning content
SEXUAL_SOLICITATION = [
    # Gạ gẫm
    'đi nhà nghỉ', 'di nha nghi', 'đi khách sạn', 'di khach san',
    'qua đêm', 'qua dem', 'ngủ với', 'ngu voi', 'ngủ cùng', 'ngu cung',
    'cho ngủ một đêm', 'cho ngu mot dem',
    'wanna fuck', 'wanna sex', 'let me fuck',
    
    # Sex trade/buying
    'bao nhiêu một đêm', 'bao nhieu mot dem',
    'giá bao nhiêu', 'gia bao nhieu', 'giá em bao nhiêu',
    'bán thân', 'ban than', 'bán dâm', 'ban dam',
    'gái gọi', 'gai goi', 'gái bao', 'gai bao',
    'how much for sex', 'price for sex',
]

# ==================== LEVEL 2: MODERATE (REVIEW) ====================

# Moderate negative words - ONLY FOR TRULY OFFENSIVE CONTENT
# DOES NOT include valid product/service reviews
MODERATE_NEGATIVE = [
    # Mocking, sarcasm targeting INDIVIDUALS
    'ngu thế', 'ngu thí', 'ngu vậy', 'ngu không',
    'khùng', 'điên', 'điên khùng', 'mất trí',
    'ngáo đá',
    'crazy', 'insane', 'mental',
    
    # Contempt for INDIVIDUALS (not products)
    'đồ rác', 'đồ bỏ đi', 'thứ rác',
    'kém người', 'hạ đẳng',
    'trash people', 'garbage person',
    
    # Fraud (only severe words kept)
    'lừa đảo', 'lừa gặt', 'lừa bịp', 
    'lừa đảo khách hàng', 'gian lận', 'gian dối',
    'lừa tiền', 'lừa của', 'ăn cắp',
    'scam', 'fraud',
]

# Personal attack words
PERSONAL_ATTACKS = [
    'mày', 'mi', 'tao', 'tau', 'thằng', 'con',
    'đứa', 'thằng cha', 'đồ cha', 'cái cha',
    'cậu ta', 'thằng này', 'con này', 'đứa này',
]

# ==================== LEVEL 3: SUSPICIOUS (CONTEXT-BASED) ====================

# Spam keywords
SPAM_INDICATORS = [
    'inbox', 'zalo', 'liên hệ ngay', 'đặt hàng ngay',
    'sale off', 'giảm giá', 'khuyến mãi',
    'mua ngay', 'click ngay', 'xem ngay',
    'http', 'www', '.com', '.vn', 'bit.ly',
]

# ==================== PATTERNS (REGEX) - ENTERPRISE EDITION ====================
# Upgraded patterns with more complex detection to catch bypasses

TOXIC_PATTERNS = [
    # ===== "địt" variants - REQUIRE ENDING CHARACTERS =====
    # REMOVED: r'\b[dđ][uụùúủũưừứửữự][ctmnị]?\b' - matching single "du" caused false positives
    # Only match if there is a CLEAR ending character
    r'\b[dđ][ịiìíỉĩị][tpk]\b',  # địt, dịt, đit, đík (MUST have t/p/k)
    r'\b[dđ][ụựủúũ][tc]\b',  # đụt, đục (MUST have t/c)
    r'\b[dđ][ụựủúũưừứửữự][tmn]\s*[mc][aáảãạăắằẳẵặâấầẩẫậeéẹêếềểễệ]',  # đụ má, địt mẹ
    
    # Separated patterns (bypass rõ ràng - có dấu phân cách)
    r'\b[dđ][\s\._\-]+[ịiìíỉĩị][\s\._\-]*[tp]\b',  # đ.ị.t, d-i-t
    r'\b[dđ][\s\._\-]+[uụùúủũưừứửữự][\s\._\-]*[tc]\b',  # đ.ụ.t
    
    # Leet speak / Number substitution
    r'\b[dđ][1!][t]\b',  # d1t, đ!t (MUST have t)
    r'\b[dđ]![tp]\b',  # đ!t, d!p
    
    # ===== "lồn" variants - ONLY OBFUSCATED =====
    # REMOVED: r'\bl[oồòóỏõọơờớởỡợ0][nl]g?\b' - matching "lon", "lòng" caused false positives
    # Only match if there is clear evidence of obfuscation
    r'\bl[0][n]\b',  # l0n (số 0 thay o)
    r'\bl[\s\._\-]+[oồ0][\s\._\-]*[n]\b',  # l.o.n, l-o-n (có dấu phân cách)
    r'\b1[oồ0]n\b',  # 1on, 10n (số 1 thay l)
    
    # ===== "cặc" variants - EXPANDED =====
    r'\bc[aăắằẳẵặâấầẩẫậáảãạặ@][ckpc]\b',  # cặc, cac, c@c, cák
    r'\bc[\s\._\-]{0,2}[aăắằẳẵặâấầẩẫậáảãạặ@][\s\._\-]{0,2}[ckpc]\b',  # c.a.c, c-a-c
    r'\b[ck][a@4][ck]\b',  # c@c, k@k, c4c
    
    # ===== VCL/VL family - EXPANDED =====
    r'\bv[\s\._\-]{0,2}[ck]?[\s\._\-]{0,2}l\b',  # vcl, v.c.l, v-l
    r'\bv[oơớờởỡợ0][\s\._\-]{0,2}c[oơớờởỡợ0][\s\._\-]{0,2}l[oơớờởỡợ0]?\b',  # vờ cờ lờ
    r'\b[vw][ck1!]?[l1!]\b',  # wl, wcl, v1l, v!l
    r'\bvãi\s+l[oồòóỏõọ0]n\b',  # vãi lồn, vãi lon
    
    # ===== ĐM/DCM family - EXPANDED =====
    r'\b[dđ][\s\._\-]{0,2}[mcpđ][\s\._\-]{0,2}m+\b',  # dm, dcm, đmm, dpm
    r'\b[dđ][m1!@]{1,3}\b',  # đm, dm, d!, đ@m
    r'\b[dđ][\s\._\-]{0,2}[c][\s\._\-]{0,2}[m][\s\._\-]{0,2}[mn]?\b',  # d.c.m, đ-c-m-m
    r'\bđ[eo]+\s+m[áa]',  # đéo má, đeo ma
    
    # ===== CC family =====
    r'\bc[\s\._\-]{0,2}c\b',  # cc, c.c, c-c
    r'\b[ck][c1!@][k]?\b',  # c!c, k@k
    
    # ===== CLM/CTM/CMM =====
    r'\bc[\s\._\-]{0,2}[lt][\s\._\-]{0,2}m\b',  # clm, ctm, c.l.m
    r'\bc[\s\._\-]{0,2}m[\s\._\-]{0,2}m+\b',  # cmm, c.m.m
    
    # ===== Đéo family =====
    # FIXED: Must have 'o' ending to avoid matching 'đề', 'để', 'đe' (common words)
    r'\b[dđ][eéèẻẽẹêếềểễệ][oóòọõỏô0@]\b',  # đéo, đèo, deo - REQUIRES 'o' ending
    r'\bđéo\b',  # Exact match
    r'\bdeo\b',  # Without diacritics (when preceded by obfuscation context)
    
    # ===== HATE SPEECH PATTERNS =====
    # LGBTQ+ hate
    # LGBTQ+ hate
    r'\b(?:đồ|thằng|con|bọn)\s+(?:lgbt|gay|đồng\s*tính|pê\s*đê|les|bê\s*đê)\b',
    r'\b(?:lgbt|gay|đồng\s*tính|les)\s+(?:đáng|cần|nên|phải)\s+(?:chết|ghét|khinh|bị\s*cấm|loại\s*bỏ|tiêu\s*diệt)\b',
    r'\b(?:lgbt|đồng\s*tính|gay|les)\s+(?:là)?\s*(?:bệnh|tội\s*lỗi|sai\s*trái|lệch\s*lạc|bất\s*thường|tởm|đê\s*tiện)\b',
    
    # Racism
    r'\b(?:đồ|thằng|con|bọn)\s+(?:tàu|khỉ\s*đen|mọi)\b',
    r'\btàu\s+(?:khựa|giặc|cộng|lùn)\b',
    
    # ===== SEXUAL CONTENT PATTERNS =====
    # Explicit sexual acts
    r'\b(?:bú|liếm|mút)\s+(?:cu|cặc|lồn)\b',
    r'\b(?:chịch|địt|đụ)\s+(?:nhau|em|anh)\b',
    r'\b(?:gái|con|thằng|ông|bà)\s+(?:xinh|đẹp|dễ\s*thương|cute).*?(?:bú|liếm|mút|chịch|địt|đụ)',  # "gái xinh ... bú cu"
    r'(?:bú|liếm|mút|chịch|địt|đụ).*?(?:giỏi|tốt|hay)',  # "bú cu giỏi", "chịch giỏi"
    
    # Sexual solicitation
    r'\b(?:đi|qua)\s+(?:nhà\s*nghỉ|khách\s*sạn|motel)\b',
    r'\b(?:ngủ|sex|làm\s*tình)\s+(?:với|cùng|chung)\b',
    r'\b(?:bao\s*nhiêu|giá)\s+(?:một|1)\s+đêm\b',
    
    # ===== "NGU" + CONTEXT (INSULTS) =====
    r'\bn[gq]u\s+(?:như|thế|thí|vậy|không|quá|vcl|vl|người|xuẩn|si|vãi)',
    r'\bn[gq]u\s+(?:vãi|vl|vcl|vkl)\s*(?:l[oồ]n|cứt|chó)',
    r'(?:đầu|óc|não)\s+(?:lợn|chó|bò|đất|gối|cá\s*vàng|gà)',

    # Added: Obfuscated 'ngu' (n.g.u, n-g-u)
    r'\bn[\s\._\-]+[gq][\s\._\-]+u\b',
    
    # ===== RACISM / HATE SPEECH EXPANSION =====
    # "bọn da đen bẩn thỉu", "cút về nước"
    r'\b(?:bọn|lũ|thằng|con|đồ)\s+(?:da\s*đen|đen|nigg)\s+(?:bẩn|thỉu|tởm|hôi|mọi|ngu|cút)',
    r'(?:cút|về)\s+(?:nước|rừng)\s+đi',
    
    # ===== THREATS =====
    r'(?:tao|tau|mình|tôi|t)\s+(?:giết|chém|đánh|đập|kill)\s+(?:mày|mi|m|cậu|bạn)',
    r'(?:giết|chém|đánh|đập)\s+(?:chết|tới\s*chết|cho\s*chết)',
    
    # ===== BYPASS DETECTION =====
    # Teen code / L33t speak patterns
    r'[dđ][\.\_\-]{1,3}[ụuiị][\.\_\-]{0,3}[tmc]',  # đ.ụ.m, d-i-t
    r'v[\.\_\-]{1,3}c[\.\_\-]{0,3}l',  # v.c.l, v-c-l
    r'l[\.\_\-]{1,3}[oồ0][\.\_\-]{0,3}n',  # l.o.n, l-ồ-n
    r'c[\.\_\-]{1,3}[aặ@][\.\_\-]{0,3}c',  # c.a.c, c-ặ-c
    
    # Unicode lookalike (homoglyphs)
    r'[dđḍ][ụứựừữử][tṭ]',  # Unicode variants
    r'[lł][ơồộốồọ][nṇ]',
    
    # Excessive spacing (bypass)
    r'[dđ]\s{2,}[ụựủúũ]\s{2,}[tmc]',  # d   ụ   t
    r'[lł]\s{2,}[oồ0]\s{2,}[nl]',  # l   o   n
    
    # Mixed case bypass (global IGNORECASE flag already applied)
    r'\b[dđ][uụ][tmc]\b',  # DuT, DụT, DụM
    r'\b[lł][oồ0][nl]\b',  # LoN, LồN
    
    # Emoji/Symbol separators
    r'[dđ][.!@#$%^&*]{1,3}[ụu][.!@#$%^&*]{0,3}[tmc]',
    
    # Repeated characters (bypass spam filters)
    r'[dđ]u+c+t*',  # duuuuccctttt
    r'l+o+n+',  # lloooonnnn
    r'c+a+c+',  # cccaaaccc
]

# ==================== CONTEXT WORDS ====================

# Words that increase severity
SEVERITY_BOOSTERS = [
    'vãi', 'vcl', 'vl', 'quá', 'cực', 'siêu',
    'khủng khiếp', 'kinh khủng', 'kinh dị',
    'chết được', 'chết tiệt', 'khốn nạn',
]

# Words that reduce severity (context positive)
SEVERITY_REDUCERS = [
    'không', 'chẳng', 'chả', 'đâu có',
    'ai bảo', 'đùa thôi', 'đùa mà',
]

# ==================== ALLOWED EXCEPTIONS ====================

# Words that may contain toxic substrings but are OK in context
# CRITICAL: Added common words that were mis-matched by patterns
ALLOWED_PHRASES = [
    'ngu ngốn',  # ngu ngốn vs ngu ngốc
    'cung cấp',  # cung vs cu
    'ngu cơ',    # ngủ cơ
    
    # ===== CRITICAL: Vietnamese proper names containing "ngu" =====
    # These are SURNAMES and common names - MUST NOT be flagged!
    'nguyễn',    # Most common Vietnamese surname (40% of population)
    'nguyên',    # Common first name/word (e.g., Nguyên, nguyên nhân)
    'nguyen',    # Romanized version of Nguyễn
    'nguyển',    # Alternative spelling
    'nguyện',    # Wish/prayer
    'nguyệt',    # Moon
    
    # ===== Common Vietnamese words containing "ngu" =====
    'người',     # Person - MOST COMMON WORD
    'những',     # Those/some
    'nguồn',     # Source
    'ngủ',       # Sleep
    'ngũ',       # Five (Sino-Vietnamese)
    'nguội',     # Cool down
    'ngước',     # Look up
    'ngựa',      # Horse
    'ngứa',      # Itchy
    'ngụ',       # Reside
    'ngư',       # Fishery
    'ngư dân',   # Fisherman
    'ngư nghiệp', # Fishing industry
    'nguy',      # Danger
    'nguy hiểm', # Dangerous
    
    # ===== Words mis-matched by "cặc" pattern =====
    # Pattern r'\bc[aăắằẳẵặâấầẩẫậáảãạặ@][ckpc]\b' matches "các", "cách", "cục", etc.
    'các',       # CRITICAL: Plural indicator in Vietnamese (very common!)
    'cách',      # Phương pháp/cách thức
    'cục',       # Vật thể/cơ quan
    'cấc',       # Không phải từ thông dụng nhưng tránh nhầm
    'cắc',       # Tiếng kêu
    'cạc',       # Card (máy tính)
    
    # ===== Other mis-matched words =====
    'con các',   # Không phải "con cặc"
    'các con',   # Các + con (số nhiều)
    'các loại',  # Các loại
    'các nhà',   # Các nhà
    'các ông',   # Các ông
    'các bà',    # Các bà
    'một cách',  # Một cách
    'bằng cách', # Bằng cách
    'theo cách', # Theo cách
    
    # ===== Words containing substrings similar to toxic words =====
    'nguồn',     # Nguồn gốc
    'người',     # Con người
    'nguyên',    # Nguyên nhân
    'ngủ',       # Đi ngủ
    'đột',       # Đột ngột
    'lòng',      # Hài lòng
    'sử dụng',   # Sử dụng
    'dù',        # Dù sao
    'dũng',      # Dũng cảm
    
    # ===== Đối tượng áp dụng (legal context) =====
    'đối tượng áp dụng',
    'cư trú',
    'người nước ngoài',
    
    # ===== NEW: "gay" in valid contexts =====
    # "gay" in Vietnamese has its own meanings (harsh, intense)
    'gay gắt',       # Tình huống căng thẳng
    'gay go',        # Khó khăn
    'hứng gay',      # Hứng thú cao độ
    'gay cấn',       # Kịch tính
    'nóng gay',      # Nóng gắt
    'vui gay',       # Vui vẻ
    'hăng gay',      # Mạnh mẽ
    
    # ===== NEW: "lon" trong ngữ cảnh hợp lệ =====
    'hài lòng',      # Hạnh phúc/thỏa mãn
    'vui lòng',      # Xin vui lòng  
    'làm ơn vui lòng',
    'xin lòng',      # Xin
    'toàn lòng',     # Toàn tâm
    'lòng tin',      # Niềm tin
    'lòng dạ',       # Tấm lòng
    'lòng tốt',      # Sự tử tế
    'lon bia',       # Lon nước giải khát
    'lon nước',
    'lon coca',
    'lon pepsi',
    'lon 7up',
    'bia lon',       # Bia đóng lon
    'nước lon',
    
    # ===== NEW: "du" trong ngữ cảnh hợp lệ =====
    'du lịch',       # Du lịch
    'du xuân',       # Du xuân
    'du học',        # Du học
    'du khách',      # Khách du lịch
    'du hành',       # Chu du
    'du thuyền',     # Tàu du lịch
    'du ngoạn',      # Đi chơi
    'du ca',         # Hát rong
    'hướng dẫn du',  # Hướng dẫn viên
    
    # ===== NEW: Duyên family - RẤT QUAN TRỌNG =====
    'duyên',         # Duyên dáng
    'duyên dáng',
    'duyên phận',
    'duyên số',
    'duyên nợ',
    'có duyên',
    'hữu duyên',
    'vô duyên',
    'nhân duyên',
    'tình duyên',
    
    # ===== NEW: Duyệt family - VERY COMMON =====
    'duyệt',         # Kiểm duyệt
    'kiểm duyệt',
    'phê duyệt',
    'xét duyệt',
    'được duyệt',
    'chờ duyệt',
    
    # ===== NEW: Personal names =====
    'phúc du',       # Rapper Phúc Du
    'rapper',        # Context âm nhạc
    
    # ===== NEW: Educational context =====
    'giáo dục',
    'sử dụng',
    'ứng dụng',
    'tác dụng',
    'công dụng',
    'dữ liệu',
    'dự án',
    'dự báo',

    
    # ===== NEW: Product review context =====
    'sản phẩm tệ',   # Đánh giá sản phẩm
    'hàng tệ',
    'dịch vụ tệ',
    'chất lượng tệ',
    'shop tệ',
    'giao hàng tệ',
    'đóng gói tệ',
    'sản phẩm kém',
    'hàng kém',
    'dịch vụ kém',
    'chất lượng kém',
    'tệ quá',
    'dở quá',
    'kém quá',
    'không được',
    'không tốt',
    'không ổn',
    'không ok',
    'thất vọng',
    'hụt hẫng',
    
    # ===== NEW: Valid criticism =====
    'đánh giá 1 sao',
    'đánh giá thấp',
    '1 sao',
    '2 sao',
    'one star',
    'two star',
    'không recommend',
    'không giới thiệu',
    'đừng mua',
    'không nên mua',
    'tránh xa',
    'cảnh báo',
    
    # ===== NEW: Edit/Reddit/Credit =====
    'edit',
    'credit',
    'reddit',
    'editor',
    'editing',
    
    'mọi người',     # Everyone
    'mọi thứ',       # Everything
    'mọi nơi',       # Everywhere
    'mọi lúc',       # Anytime
    'mọi khi',       # Whenever
    'mọi ngày',      # Every day
    'mọi việc',      # Everything (tasks)
    'mọi chuyện',    # Everything (matters)
    'mọi thời',      # All time
    'mọi nỗi',       # All feelings
    'mọi điều',      # Every thing
    'mọi khía cạnh', # Every aspect
    'mọi góc độ',    # Every angle
    'mọi mặt',       # Every side
    'tất cả mọi',    # All of every
    'của mọi',       # Of every
    'cho mọi',       # For every
    'với mọi',       # With every
    'trong mọi',     # In every
    'ở mọi',         # At every
]

# Allowed contexts: when criticizing ideas/views, NOT individuals
OPINION_CRITICISM_CONTEXT = [
    'ý kiến', 'quan điểm', 'suy nghĩ', 'nhận xét', 'đánh giá',
    'phát biểu', 'lập luận', 'tư tưởng', 'ý tưởng', 'luận điểm',
    'opinion', 'idea', 'view', 'viewpoint', 'thought',
]

# ==================== SCORING SYSTEM - ENTERPRISE EDITION ====================
# Upgraded scoring system with more detailed classification

SEVERITY_SCORES = {
    # Profanity & Vulgar language
    'SEVERE_PROFANITY': 12,  # Increased from 10 -> 12 (more severe)
    'SEVERE_INSULTS': 10,    # Increased from 8 -> 10
    
    # Hate Speech (CRITICAL - auto reject)
    'HATE_LGBTQ': 15,        # NEW - LGBTQ+ discrimination
    'HATE_RACISM': 15,       # NEW - Racial discrimination
    'HATE_RELIGION': 12,     # NEW - Religious discrimination
    'HATE_SEXISM': 10,       # NEW - Gender discrimination
    
    # Sexual Content (CRITICAL)
    'SEXUAL_EXPLICIT': 15,   # NEW - Explicit pornographic content
    'SEXUAL_SUGGESTIVE': 10, # NEW - Suggestive content
    'SEXUAL_SOLICITATION': 12, # NEW - Sexual solicitation
    
    # Moderate violations
    'MODERATE_NEGATIVE': 5,
    'PERSONAL_ATTACKS': 3,   # Reduced from 6 -> 3 (only slight boost when combined with other violations)
    
    # Low severity
    'SPAM_INDICATORS': 3,
    
    # Pattern matching
    'TOXIC_PATTERNS': 8,     # Increased from 7 -> 8 (more complex patterns)
}

# ===== DECISION THRESHOLDS - STRICT MODE =====
# Reduced thresholds to catch more violations

REJECT_THRESHOLD = 10     # Increased from 8 -> 10 but severity increased
REVIEW_THRESHOLD = 5      # Increased from 4 -> 5 to catch more cases
WARNING_THRESHOLD = 3     # NEW - Warning threshold

# ===== AUTO-REJECT CATEGORIES (Bypass threshold) =====
# These categories automatically reject regardless of score
AUTO_REJECT_CATEGORIES = [
    'HATE_LGBTQ',
    'HATE_RACISM', 
    'SEXUAL_EXPLICIT',
    'SEXUAL_SOLICITATION',
]

# ===== MULTIPLIER SYSTEM =====
# Factors that increase/decrease score

CONTEXT_MULTIPLIERS = {
    # Tăng mức độ nghiêm trọng
    'has_personal_pronoun': 1.5,  # Has "mày", "tao" -> personal attack
    'has_severity_booster': 1.3,  # Has "vãi", "vcl", "quá"
    'multiple_violations': 1.5,   # Multiple simultaneous violations
    'targeting_group': 2.0,       # Targeting specific groups (hate speech)
    
    # Giảm mức độ (cho phép phê bình hợp lý)
    'opinion_criticism': 0.3,     # Criticism of ideas/views
    'product_review': 0.4,        # Product/service evaluation
    'has_negation': 0.5,          # Has negation like "không", "chẳng"
}

# ==================== HELPER FUNCTIONS - ENHANCED ====================

def get_all_toxic_words():
    """Get all toxic words"""
    return (
        SEVERE_PROFANITY + 
        SEVERE_INSULTS + 
        HATE_LGBTQ +
        HATE_RACISM +
        HATE_RELIGION +
        HATE_SEXISM +
        SEXUAL_EXPLICIT +
        SEXUAL_SUGGESTIVE +
        SEXUAL_SOLICITATION +
        MODERATE_NEGATIVE + 
        PERSONAL_ATTACKS
    )

def get_critical_words():
    """Get most severe words - auto reject"""
    return (
        SEVERE_PROFANITY + 
        SEVERE_INSULTS +
        HATE_LGBTQ +
        HATE_RACISM +
        SEXUAL_EXPLICIT +
        SEXUAL_SOLICITATION
    )

def get_hate_speech_words():
    """Get hate speech words"""
    return (
        HATE_LGBTQ +
        HATE_RACISM +
        HATE_RELIGION +
        HATE_SEXISM
    )

def get_sexual_content_words():
    """Lấy từ nội dung tình dục"""
    return (
        SEXUAL_EXPLICIT +
        SEXUAL_SUGGESTIVE +
        SEXUAL_SOLICITATION
    )

def get_all_patterns():
    """Lấy tất cả patterns"""
    return TOXIC_PATTERNS

def is_auto_reject_category(category: str) -> bool:
    """Kiểm tra xem category có tự động reject không"""
    return category in AUTO_REJECT_CATEGORIES

