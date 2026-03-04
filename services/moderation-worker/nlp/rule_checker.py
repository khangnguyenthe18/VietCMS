

import re
from typing import Dict, List, Tuple, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


# ==================== STEM-BASED TOXIC PATTERNS ====================
# Using regex stems to catch variations

# Core profanity stems (Vietnamese)
PROFANITY_STEMS = {
    # ĐỤ/ĐỊT family
    'dit': {
        'patterns': [
            r'\bđ[ịiìíỉĩ]t\b',
            r'\bd[ịiìíỉĩ]t\b',
            r'\bdjt\b',
            r'\bđjt\b',
            r'\bd1t\b',
            r'\bđ1t\b',
            r'\bd!t\b',
            r'\bđ!t\b',
            # địt mẹ/má patterns
            r'\bđ[ịiìíỉĩ]t\s+m[ẹeèéẻẽẹ]',  # địt mẹ
            r'\bđ[ịiìíỉĩ]t\s+m[áaàảãạ]',   # địt má
        ],
        'stripped_pattern': r'\bdit\b',  # For no-diacritics version
        'severity': 'severe',
        'labels': ['toxicity', 'profanity'],
    },
    
    # ĐM/DCM family  
    'dm': {
        'patterns': [
            r'\bđm+\b',
            r'\bdm+\b',
            r'\bđcm+\b',
            r'\bdcm+\b',
            r'\bđkm+\b',
            r'\bdkm+\b',
            r'\bđ[ụu]\s*m[áaẹe]',  # đụ má, đụ mẹ
            r'\bd[ịi]t\s*m[áaẹe]', # địt má, địt mẹ
        ],
        'stripped_pattern': r'\bdm+\b',
        'severity': 'severe',
        'labels': ['toxicity', 'profanity'],
    },
    
    # LỒN family
    'lon': {
        'patterns': [
            r'\bl[ồôoòóỏõọ]n\b',
            r'\bl0n\b',
            r'\b1on\b',
            r'\b10n\b',
        ],
        'stripped_pattern': r'\blon\b',
        'severity': 'severe',
        'labels': ['toxicity', 'profanity'],
        # Safe contexts where "lon" is OK
        'safe_contexts': [
            'lon bia', 'bia lon', 'lon nước', 'nước lon',
            'lon coca', 'lon pepsi', 'lon 7up', 'lon redbull',
            'hài lòng', 'vui lòng', 'lòng tin', 'lòng tốt',
            'tấm lòng', 'toàn lòng', 'xin lòng',
        ],
    },
    
    # CẶC family
    'cac': {
        'patterns': [
            r'\bc[ặăắằẳẵạa]c\b',
            r'\bc@c\b',
            r'\bc4c\b',
            r'\bkac\b',
            r'\bk[ặăa]c\b',
        ],
        'stripped_pattern': r'\bcac\b',
        'severity': 'severe',
        'labels': ['toxicity', 'profanity'],
        'safe_contexts': [
            'các bạn', 'các anh', 'các chị', 'các em', 'các bác',
            'các ông', 'các bà', 'các cháu', 'các con',
            'một cách', 'bằng cách', 'theo cách', 'có cách',
            'các loại', 'các kiểu', 'các dạng',
        ],
    },
    
    # VCL/VL family
    'vcl': {
        'patterns': [
            r'\bvcl\b',
            r'\bvkl\b',
            r'\bvl\b',
            r'\bvãi\s*l[ồôo]n',
            r'\bvai\s*lon\b',
            r'\bvờ\s*cờ\s*lờ\b',
        ],
        'stripped_pattern': r'\b(vcl|vkl|vl)\b',
        'severity': 'severe',
        'labels': ['toxicity', 'profanity'],
    },
    
    # CC family (con/cái cặc)
    'cc': {
        'patterns': [
            r'\bcc\b',
            r'\bcờ\s*cờ\b',
        ],
        'stripped_pattern': r'\bcc\b',
        'severity': 'moderate',
        'labels': ['toxicity', 'profanity'],
    },
    
    # CLM/CTM family
    'clm': {
        'patterns': [
            r'\bclm\b',
            r'\bctm\b',
            r'\bcmm\b',
        ],
        'stripped_pattern': r'\b(clm|ctm|cmm)\b',
        'severity': 'severe',
        'labels': ['toxicity', 'profanity'],
    },
    
    # NGU family (context-dependent)
    # NGU family (context-dependent)
    'ngu': {
        'patterns': [
            r'\bngu\s+(như|thế|thí|vậy|quá|vãi|vcl|vl|vkl|hơn)',
            r'\bngu\s+ngốc\b',
            r'\bngu\s+si\b',
            r'\bngu\s+xuẩn\b',
            r'\bngu\s+(như)?\s*(bò|chó|lợn|heo|vật)',
        ],
        'stripped_pattern': r'\bngu\s+(nhu|the|thi|vay|qua|ngoc|si|xuan|bo|cho|lon)\b',
        'severity': 'moderate',
        'labels': ['insult'],
        'safe_contexts': [
            'ngủ', 'nguồn', 'người', 'nguyên', 'nguyễn', 'nguyen',
            'nguội', 'ngước', 'ngựa', 'ngứa', 'ngư dân', 'nguyện', 'nguyệt',
            'nguỵ', 'ngụy', 'những', 'nguy', 'nguy hiểm', 'ngư nghiệp',
        ],
        'context_required': True,
        # Expanded patterns for "ngu"
        'additional_patterns': [
             # Target + ngu
             r'\b(ý\s*kiến|quan\s*điểm|bài\s*viết|post|tus|stt|cmt|comment|nhận\s*xét)\s+ngu\b',
             r'\b(nói|phát\s*biểu|comment|trả\s*lời)\s+ngu\b',
             r'\b(mày|mi|nó|thằng|con)\s+(này|kia|đó)?\s*(cũng|thì|mà)?\s*ngu\b',
             # Ngu + intensifier
             r'\bngu\s+(vãi|thật|thế|chứ|hơn|hết|ko\s*tả|bỏ\s*mẹ|bà\s*cố|vcc|vc)\b',
             r'\bngu\s+như\s+(bò|chó|lợn|heo|thú)\b',
        ]
    },
    
    # NEW: DEO/DELL family
    'deo': {
        'patterns': [
            r'\bđéo\b',
            r'\bđ[éèe]o\b',
            r'\bđé[0o]\b',
            r'\bd[eé]o\b',
            r'\bđell\b',
            r'\bdell\b',
            r'\bdeo\b', # Explicitly add no-accent version to main patterns
        ],
        'stripped_pattern': r'\b(deo|dell)\b',
        'severity': 'severe',
        'labels': ['profanity', 'toxicity'],
    },
    
    # NEW: CON ME MAY / CON CAI family
    'parent_insult': {
        'patterns': [
            r'\bcon\s*(mẹ|má|cặc|cac)\s*(mày|mi|nó)\b',
            r'\bcái\s*(địt|đm|đkm|đcm)\b',
            r'\bmẹ\s*kiếp\b',
            r'\bcon\s*chó\b',
        ],
        'severity': 'severe',
        'labels': ['profanity', 'insult'],
    },

    # NGU bypass - Explicitly catch n.g.u, n g u, nguuu 
    # These forms are intentional bypasses, so NO safe context filtering needed
    'ngu_bypass': {
        'patterns': [
            r'\bn[\s\._\-]+g[\s\._\-]+u\b',  # n.g.u, n-g-u, n g u (MUST have separator)
            r'\bngu{2,}\b',                  # nguu, nguuu (Testing repeated chars)
            r'\bng+[kq]u\b',                 # ngku, ngqu (Teencode mix)
            r'\bn\s*g\s*u\b',                # n g u (spaced)
        ],
        'severity': 'moderate',
        'labels': ['insult', 'obfuscation'],
    },
    
    # Brain/Head insults (standalone patterns)
    'brain_insults': {
        'patterns': [
            r'\bnão\s+(lợn|chó|bò|gà|cá\s*vàng|gối|đất|ngắn|phẳng)\b',
            r'\bóc\s+(lợn|chó|bò|gà|cá\s*vàng|gối|đất|chim|ngắn)\b',
            r'\bđầu\s+(lợn|chó|bò|gà|gối|đất|bò|cá|tôm)\b',
            r'\bkhông\s*có\s*não\b',
        ],
        'severity': 'moderate',
        'labels': ['insult'],
    },
    
    # NEW: Trash/Garbage insults
    'garbage': {
        'patterns': [
            r'\b(như|là)\s*rác\b',
            r'\brác\s*rưởi\b',
            r'\bđồ\s*rác\b',
            r'\b(nói|viết|làm)\s*(gì)?\s*cũng\s*rác\b',
        ],
        'severity': 'moderate',
        'labels': ['insult'],
        'context_required': True, 
         'additional_patterns': [
             r'\b(phim|video|bài|post|tus|ý\s*kiến|nội\s*dung)\s*(này|đó|kia)?\s*(như|là)?\s*rác\b',
             r'\b(kênh|page|group)\s*(này|đó)?\s*rác\b',
         ]
    },
    
    # Standalone insults (only flag when obfuscated)
    'obfuscated_insults': {
        'patterns': [
            # These are flagged ONLY when obfuscation is detected
            # Normal "ngu" standalone is not flagged
            # But "n.g.u" or "n-g-u" signals intentional bypass
        ],
        'standalone_words': ['ngu', 'ngốc', 'điên', 'khùng', 'dở', 'cút', 'biến'],  # Special handling
        'severity': 'moderate',
        'labels': ['insult', 'obfuscation_bypass'],
        'only_when_obfuscated': True,  # Key flag
    },
}


# ==================== HARASSMENT / BODY-SHAMING ====================
# These are NOT profane but still harmful when targeting a person

HARASSMENT_PATTERNS = {
    # Body-shaming / Appearance attacks
    'appearance_attack': {
        'patterns': [
            # Direct insults about appearance
            r'\b(mày|mi|nó|đứa\s*này|thằng\s*này|con\s*này)\s+(xấu|xí|bẩn|ghê|kinh|tởm|gớm)',
            r'\b(mặt|da|người|thân|body)\s+(mày|mi|nó)\s+(xấu|bẩn|ghê|kinh)',
            r'\b(xấu|xí|bẩn|ghê|kinh|tởm)\s+(quá|thế|vậy|quá\s*trời|vãi)',
            
            # "nhìn mặt mày... muốn nôn" pattern
            r'\bnhìn\s+(mặt|mày|mi|nó).*?(muốn\s*nôn|ghê\s*tởm|kinh\s*tởm|ớn|ghét)',
            
            # "mày/mi xấu..." direct attack
            r'\b(sao\s+)?(mày|mi|nó)\s+(xấu|xí|bẩn|hôi|thối|dơ)',
        ],
        'severity': 'moderate',
        'labels': ['harassment', 'body_shaming'],
        'requires_target': True,  # Must target a person (mày/mi/nó)
    },
    
    # Personal attack indicators
    'personal_attack': {
        'patterns': [
            # "đồ X" pattern (đồ ngu, đồ khốn, đồ chó...)
            r'\bđồ\s+(ngu|ngốc|khốn|chó|lợn|bò|súc\s*vật|rác|vô\s*dụng|hèn)',
            
            # "thằng/con X" pattern
            r'\b(thằng|con)\s+(ngu|ngốc|khốn|chó|lợn|điên|khùng|rồ|dở)',
            
            # "thằng/con này ngu" pattern
            r'\b(thằng|con)\s+(này|đó|kia)\s+(ngu|ngốc|khốn|điên)',
            
            # "mày là đồ X"
            r'\b(mày|mi|nó)\s+(là\s+)?(đồ|thằng|con)\s+(ngu|ngốc|khốn|chó)',
        ],
        'severity': 'moderate',
        'labels': ['harassment', 'insult'],
        'requires_target': False,  # These patterns inherently indicate targeting
    },
    
    # Contempt expressions
    'contempt': {
        'patterns': [
            r'\b(ghét|khinh|tởm|gớm|ớn|chán)\s+(mày|mi|nó|bọn\s*này)',
            r'\b(mày|mi|nó).*?(đáng\s*khinh|đáng\s*ghét|đáng\s*chết)',
            r'\b(vô\s*dụng|vô\s*giá\s*trị|không\s*ra\s*gì)\s*$',
        ],
        'severity': 'moderate',
        'labels': ['harassment'],
        'requires_target': True,
    },
    
    # NEW: Doxxing / Threatening to release info
    'doxxing': {
        'patterns': [
            # Flexible pattern: Verb ... Info ... Target
            r'\b(tung|đăng|public|lộ|share).*?(địa\s*chỉ|sdt|số\s*điện\s*thoại|nhà|clip|ảnh|thông\s*tin).*?(của)?\s*(mày|mi|nó)\b',
            r'\b(cho|để)\s*(mọi\s*người|cả\s*làng|ai\s*cũng)\s*(biết|xem)\s*(mặt|nhà|sdt).*?(mày|mi)\b',
            r'\b(công\s*khai)\s*(thông\s*tin|địa\s*chỉ|sdt|danh\s*tính).*?(mày|mi|nó)\b',
            r'\b(biết|có)\s*(địa\s*chỉ|nhà|sdt)\s*(mày|mi|nó).*?(rồi)\b',
        ],
        'severity': 'severe',
        'labels': ['harassment', 'doxxing', 'threat'],
        'requires_target': True,
    },
    
    # NEW: Sexual Harassment
    'sexual_harassment': {
        'patterns': [
            r'\b(vú|ngực|mông|body|hàng)\s*(ngon|to|đẹp|thơm|múp)\b',
            r'\b(làm|chịch|xoạc|quan\s*hệ)\s*(tình\s*dục|nháy|phát)\b',
            r'\b(sướng|phê)\s*(lắm|vãi)\b',
            r'\b(gái|em)\s*(xinh|ngon).*?(giỏi|chắc|thế|quá|vậy).*?(lắm|nhỉ|cơ|chứ)',
            r'\b(gái|em)\s*(xinh|ngon).*?(làm|chịch|xoạc)\b',
            r'\b(nhìn|xem)\s*(cái\s*đó|hàng|sex|xxx)\b',
        ],
        'severity': 'severe',
        'labels': ['harassment', 'sexual_harassment'],
    },
}


# ==================== SPAM / SCAM PATTERNS ====================
SPAM_PATTERNS = {
    'make_money': {
        'patterns': [
            r'\bkiếm\s*tiền\s*(online|tại\s*nhà|nhanh|trên\s*mạng)\b',
            r'\b(thu\s*nhập|lương)\s*(\d+|hàng)\s*(triệu|tr|k|củ)\s*/?\s*(ngày|tháng|tuần|h)\b',
            r'\b(việc|làm)\s*nhẹ\s*lương\s*cao\b',
            r'\b(vốn|bỏ\s*ra)\s*0\s*(đ|dong|đồng)\b',
            r'\bkhông\s*cần\s*(kinh\s*nghiệm|bằng\s*cấp)\b',
        ],
        'severity': 'severe',  # Spam is usually auto-reject/hide
        'labels': ['spam', 'scam'],
    },
    'recruitment': {
        'patterns': [
            r'\btuyển\s*(ctv|cộng\s*tác\s*viên|sỉ|lẻ|đại\s*lý)\b',
            r'\b(ib|inbox|nhắn\s*tin)\s*(ngay|cho|mình|em|tôi)\b',
            r'\b(call|liên\s*hệ|zalo|tele|telegram)\s*[:.]*\s*0\d{9}\b',
            r'\b(kết\s*bạn|kb|add)\s*(zalo|tele)',
        ],
        'severity': 'severe',
        'labels': ['spam', 'ads'],
    },
    'gambling': {
        'patterns': [
            r'\b(nhà\s*cái|bóng\s*đá|lô\s*đề|xổ\s*số|tài\s*xỉu|nổ\s*hũ|bắn\s*cá)\b',
            r'\b(soi\s*kèo|chốt\s*số|bạch\s*thủ)\b',
            r'\b(bet|casino|game\s*bài)\b',
        ],
        'severity': 'severe',
        'labels': ['spam', 'gambling'],
    },
    # NEW: Health/Weight loss scams
    'health_scam': {
        'patterns': [
            r'\b(giảm\s*cân|tăng\s*cân)\s*(siêu\s*tốc|nhanh|không\s*cần)\b',
            r'\b(tăng\s*chiều\s*cao|cao\s*thêm)\s*\d+\s*(cm|phân)\b',
            r'\b(không\s*cần)\s*(ăn\s*kiêng|tập\s*luyện|thể\s*dục)\b',
            r'\b(mua\s*ngay|đặt\s*ngay|order\s*ngay)\s*[!💊💰]+',
            r'[💊🔥]{2,}',  # Multiple medicine/fire emojis = likely spam
        ],
        'severity': 'severe',
        'labels': ['spam', 'health_scam'],
    },
    # NEW: Crypto/Investment scams  
    'crypto_scam': {
        'patterns': [
            r'\b(đầu\s*tư)\s*(bitcoin|crypto|btc|eth)\b',
            r'\b(lãi|lợi\s*nhuận)\s*\d+\s*%\s*/\s*(ngày|tháng|tuần)\b',
            r'\b(sale\s*sốc|giảm\s*giá)\s*\d+\s*%\b',
            r'[💰💎🚀]{2,}',  # Multiple money emojis = likely spam
        ],
        'severity': 'severe',
        'labels': ['spam', 'crypto_scam'],
    },
    # NEW: Hacking/Illegal services
    'illegal_services': {
        'patterns': [
            r'\b(hack)\s*(acc|account|tài\s*khoản|fb|facebook|zalo)\b',
            r'\b(bẻ\s*khóa|crack)\s*(phần\s*mềm|game|acc)\b',
            r'\b(mua\s*bán)\s*(acc|account|tài\s*khoản)\b',
            r'\b(liên\s*hệ|contact)\s*(ngay|nhanh|gấp)!*\b',
        ],
        'severity': 'severe',
        'labels': ['spam', 'illegal'],
    },
    # NEW: Adult recruitment/services
    'adult_services': {
        'patterns': [
            r'\b(cần|tuyển)\s*(gái|girl)\s*(xinh|đẹp|dịch\s*vụ)\b',
            r'\b(gái|girl)\s*(xinh|đẹp).*?(dịch\s*vụ|lương\s*cao)\b',
            r'\b(dịch\s*vụ)\s*(đặc\s*biệt|vip|người\s*lớn)\b',
            r'\b(ảnh\s*nóng|clip\s*nóng|video\s*nóng)\b',
            r'\b(xem|tải)\s*(ảnh|clip|video)\s*(nóng|18\+|xxx)\b',
        ],
        'severity': 'severe',
        'labels': ['spam', 'adult'],
    },
    # NEW: Prize/Lottery Scams - "Bạn trúng 100 triệu! Click để nhận"
    'prize_scam': {
        'patterns': [
            # Prize winning patterns
            r'\b(trúng|được\s*tặng|nhận\s*ngay)\s*\d+\s*(triệu|tỷ|k|tr|đ|usd|vnd)\b',
            r'\b(trúng|thắng)\s*(giải|thưởng|jackpot)\b',
            r'\b(chúc\s*mừng|xin\s*chúc\s*mừng).*?(trúng|thắng|được)\b',
            r'\bclick\s*(để|vào|link|ngay|nhận)\b',
            r'\b(nhấn|bấm|ấn)\s*(vào|link|đây|ngay)\s*(để\s*)?(nhận|lấy)\b',
            # URL patterns combined with scam indicators
            r'(http[s]?://|www\.)[^\s]+.*(nhận|lấy|trúng|thưởng)',
            r'(trúng|thưởng|tặng).*(http[s]?://|www\.)',
        ],
        'severity': 'severe',
        'labels': ['spam', 'scam', 'phishing'],
    },
    # NEW: Phishing URLs and suspicious links
    'phishing_url': {
        'patterns': [
            # Common phishing domain patterns
            r'(http[s]?://|www\.)[^\s]*\.(vn|com|net|org)[^\s]*\s*(click|nhận|đăng\s*ký|login)',
            # Short link with scam context
            r'(bit\.ly|tinyurl|t\.co|goo\.gl)[^\s]*',
            # Suspicious URL keywords
            r'(http[s]?://)[^\s]*(prize|reward|lucky|winner|bonus|free-?gift)[^\s]*',
        ],
        'severity': 'severe',
        'labels': ['spam', 'phishing'],
    },
    # NEW: Solicitation / Sexual solicitation - "Nhắn tin cho anh, anh cho xem cái hay"
    'solicitation': {
        'patterns': [
            # "Nhắn tin cho anh/em + show something"
            r'\b(nhắn\s*tin|inbox|ib|dm)\s*(cho)?\s*(anh|em|tôi|mình).*?(cho\s*xem|xem|tặng|gửi)\b',
            r'\b(anh|em)\s*(cho|gửi|tặng)\s*(xem|link|ảnh|hình|video)\b',
            # Contact request with implicit content
            r'\b(add|kết\s*bạn|kb)\s*(zalo|tele|fb).*?(xem|hay|hot|18\+|xxx)\b',
            # "Xem cái hay/hot"
            r'\b(xem|coi)\s*(cái)?\s*(hay|hot|đặc\s*biệt|18\+|xxx|nóng)\b',
            # Follow patterns
            r'\b(follow|theo\s*dõi)\s*(để\s*)?(nhận|xem|có)\b',
        ],
        'severity': 'severe',
        'labels': ['spam', 'solicitation'],
    },
    # NEW: Engagement bait (less severe, for review)
    'engagement_bait': {
        'patterns': [
            # Like/share bait
            r'\b(like|share|chia\s*sẻ)\s*(để|cho|giúp)\s*(ủng\s*hộ|nhiều\s*người|mọi\s*người)\b',
            r'\b(like|sub|đăng\s*ký)\s*(kênh|page|channel)\b',
            # Comment farming
            r'\b(comment|bình\s*luận)\s*(số|tên|tuổi)\s*(để|của)\s*(bạn|mình)\b',
        ],
        'severity': 'moderate',  # Less severe, just review
        'labels': ['spam', 'engagement_bait'],
    },
}


# ==================== HATE SPEECH PATTERNS ====================
# Discrimination based on race, ethnicity, nationality, gender, religion, etc.

HATE_SPEECH_PATTERNS = {
    # Racial discrimination
    'racism': {
        'patterns': [
            # Anti-black
            r'\b(bọn|lũ|đám|thằng|con)\s*(da\s*đen|đen|mọi)\b',
            r'\b(da\s*đen|người\s*đen).*?(bẩn|thối|xấu|ghê|cút|về\s*nước)',
            r'\b(cút|biến|đi\s*chỗ\s*khác|về\s*nước).*?(da\s*đen|đen)',
            r'\bkhỉ\s*đen\b',
            r'\bmọi\s*đen\b',
            # General racism
            r'\b(châu\s*phi|người\s*châu\s*phi)\s*(kém|văn\s*minh|lạc\s*hậu|ngu)\b',
            r'\b(kém|thiếu)\s*văn\s*minh\b',
            
            # Anti-Chinese
            r'\b(bọn|lũ|đám|thằng)\s*tàu\s*(khựa|cộng|giặc)?\b',
            r'\btàu\s*(khựa|cộng|giặc)\b',
            r'\b(chink|ching\s*chong)\b',
            
            # Anti-minority
            r'\b(bọn|lũ|đám)\s*(mọi|thổ\s*dân|rừng\s*núi)\b',
            r'\b(dân\s*tộc|miền\s*núi|thiểu\s*số).*?(ngu|dốt|lạc\s*hậu|bẩn|nghèo\s*nàn)',
        ],
        'severity': 'severe',
        'labels': ['hate', 'racism'],
    },
    
    # LGBTQ+ discrimination
    'lgbtq_hate': {
        'patterns': [
            r'\b(đồ|thằng|con|bọn)\s*(gay|đồng\s*tính|pê\s*đê|bê\s*đê|les)',
            r'\b(gay|đồng\s*tính).*?(bệnh|đáng\s*chết|tởm|ghê|kinh)',
            r'\b(tiêu\s*diệt|giết|đánh)\s*(gay|đồng\s*tính|pê\s*đê)',
            # NEW: More LGBT hate patterns
            r'\b(lgbt|gay|les)\s*(là)?\s*(tội\s*lỗi|bệnh|loạn|sai\s*trái)',
            r'\b(chuyển\s*giới|trans)\s*(là)?\s*(loạn|bệnh|điên|thần)',
            r'\b(đồng\s*tính|gay)\s*(đáng)?\s*(bị)?\s*(khinh|ghét|cấm)\b',
            r'\blgbt.*?(nên|cần|phải)\s*(bị)?\s*(cấm|loại\s*bỏ|tiêu\s*diệt)\b',
            r'\b(khinh\s*bỉ|ghê\s*tởm)\s*(gay|đồng\s*tính|lgbt)\b',
            r'\b(người\s*)?(đồng\s*tính|gay)\s*(là)?\s*(bệnh\s*tâm\s*thần|tâm\s*thần)',
        ],
        'severity': 'severe',
        'labels': ['hate', 'lgbtq_discrimination'],
    },
    
    # NEW: Sexism / Gender discrimination
    'sexism': {
        'patterns': [
            # Anti-women
            r'\b(đàn\s*bà|phụ\s*nữ|con\s*gái)\s*(ngu|dốt|không\s*có\s*não|không\s*biết\s*gì)',
            r'\b(phụ\s*nữ|con\s*gái)\s*(không\s*có|thiếu|không\s*đủ)\s*(não|thông\s*minh|logic)',
            r'\b(chỉ|chỉ\s*có)\s*(đàn\s*ông|nam)\s*(mới)?\s*(có|biết)\s*(logic|tư\s*duy)',
            r'\b(đàn\s*bà|phụ\s*nữ).*?(đừng|không\s*nên)\s*(học|làm|cãi)',
            r'\b(con\s*gái|phụ\s*nữ)\s*(không\s*nên|đừng)\s*(học|làm)\s*(it|kỹ\s*thuật|công\s*nghệ)',
            r'\b(phụ\s*nữ|đàn\s*bà)\s*(thì)?\s*(không)\b',
        ],
        'severity': 'severe',
        'labels': ['hate', 'sexism'],
    },
    
    # NEW: Religion discrimination
    'religion_hate': {
        'patterns': [
            # General religion hate
            r'\b(tôn\s*giáo|đạo)\s*(x|y|z)?\s*(toàn)?\s*(khủng\s*bố|cực\s*đoan|điên|lạc\s*hậu)',
            r'\b(người\s*theo\s*đạo|tín\s*đồ)\s*(x|y|z)?\s*(lạc\s*hậu|ngu|dốt|điên)',
            r'\b(vô\s*thần)\s*(không\s*có|thiếu)\s*(đạo\s*đức|lương\s*tâm)',
            r'\b(phật\s*giáo|thiên\s*chúa|hồi\s*giáo|tin\s*lành).*?(ngu|dốt|khủng\s*bố|cực\s*đoan)',
            r'\b(người\s*)?(vô\s*thần|không\s*theo\s*đạo).*?(không\s*có\s*đạo\s*đức|vô\s*đạo\s*đức)',
        ],
        'severity': 'severe',
        'labels': ['hate', 'religion_discrimination'],
    },
    
    # Xenophobia
    'xenophobia': {
        'patterns': [
            r'\b(cút|biến|đi|về)\s*(về\s*nước|đi\s*chỗ\s*khác|khỏi\s*đây)',
            r'\b(ngoại\s*quốc|người\s*nước\s*ngoài|dân\s*nhập\s*cư).*?(cút|biến|về|bẩn)',
            # "biến đi (người nước ngoài/ngoại quốc)"
            r'\b(biến|cút)\s+(đi\s+)?(người\s*nước\s*ngoài|ngoại\s*quốc|dân\s*nhập\s*cư)',
        ],
        'severity': 'moderate',
        'labels': ['hate', 'xenophobia'],
        # NOTE: Removed additional_context - these patterns are already specific
    },
}


# ==================== THREAT / VIOLENCE PATTERNS ====================
# Direct threats of violence, harm, or intimidation

THREAT_PATTERNS = {
    # Direct physical threats - "Tao sẽ tìm mày và đánh cho chừa"
    'physical_threat': {
        'patterns': [
            # "tìm mày và đánh/giết"
            r'\b(tao|tôi|anh|bọn\s*tao)\s*(sẽ|phải)\s*(tìm|kiếm|đến)\s*(mày|mi|nó|nhà\s*mày).*?(đánh|giết|đập|bóp|chém|đâm)\b',
            r'\b(tao|tôi)\s*(sẽ)?\s*(đánh|giết|đập|chém|đâm|bóp\s*cổ)\s*(mày|mi|nó|chết)\b',
            # "đánh cho chừa/chết"
            r'\b(đánh|đập|chém|giết)\s*(cho)?\s*(chừa|chết|tan\s*xương|gãy\s*tay|nát\s*mặt)\b',
            # Violence verbs with targeting
            r'\b(mày|mi|nó)\s*(sẽ|phải)?\s*(chết|đánh|ăn\s*đòn)\b',
            r'\b(sẽ|phải)\s*(giết|đánh|bóp\s*cổ|chém)\s*(mày|mi|nó|hết)\b',
        ],
        'severity': 'severe',
        'labels': ['threat', 'violence'],
    },
    # Death threats
    'death_threat': {
        'patterns': [
            r'\b(mày|mi|nó)\s*(phải|sẽ|đáng)\s*(chết|chết\s*đi|đi\s*chết)\b',
            r'\b(chết|chết\s*đi|chết\s*mẹ\s*đi)\s+(mày|mi|nó|đi)\b',
            r'\b(giết|giết\s*chết|bóp\s*cổ|chém\s*chết)\s*(mày|mi|nó|hết|cả\s*nhà)\b',
            r'\b(tao|bọn\s*tao)\s*(sẽ)?\s*(giết|chém|xử)\s*(mày|mi|nó|chết)\b',
        ],
        'severity': 'severe',
        'labels': ['threat', 'violence', 'death_threat'],
    },
    # Intimidation / stalking
    'intimidation': {
        'patterns': [
            r'\b(tao\s+)?(biết|tìm\s*được)\s*(nhà|địa\s*chỉ|chỗ\s*ở)\s*(của\s*)?(mày|mi|nó)\b',
            r'\b(tao|bố)\s*(có|biết|nắm)\s*(sdt|số|địa\s*chỉ|nhà|thông\s*tin)\s*(mày|mi|nó)\b',
            r'\b(liệu\s*hồn|coi\s*chừng|đợi\s*đấy|chờ\s*đấy|chờ\s*đó)\b',
            r'\b(mày|mi)\s*(liệu|đợi|chờ)\s*(đấy|đó|hồn)\b',
            r'\b(sẽ|phải)\s*(biết\s*tay|hối\s*hận)\b',
            r'\b(đến|tới|qua|tìm)\s*(nhà|chỗ)\s*(mày|mi|nó)\b',
        ],
        'severity': 'moderate',
        'labels': ['threat', 'intimidation'],
    },
}


# ==================== PII (PERSONAL IDENTIFIABLE INFORMATION) PATTERNS ====================
# Bank accounts, phone numbers, addresses, ID numbers

PII_PATTERNS = {
    # Bank account numbers
    'bank_account': {
        'patterns': [
            # STK/Số tài khoản patterns
            r'\b(stk|số\s*tài\s*khoản|tk)\s*[:.]*\s*\d{8,20}\b',
            r'\b(vietcombank|techcombank|mbbank|tpbank|acb|bidv|agribank|vpbank|sacombank)\s*[:.]*\s*\d{8,20}\b',
            # Bank name + account
            r'\b(ngân\s*hàng|nh)\s*[:.]*\s*\w+\s*\d{8,20}\b',
        ],
        'severity': 'moderate',
        'labels': ['pii', 'bank_account'],
    },
    # Phone numbers (Vietnamese format)
    'phone_number': {
        'patterns': [
            # Vietnamese phone format with context
            r'\b(sdt|số\s*điện\s*thoại|liên\s*hệ|hotline|zalo|tele)\s*[:.]*\s*0[0-9]{9,10}\b',
            # Standalone phone number (10 digits starting with 0)
            r'\b0[35789][0-9]{8}\b',
            r'\b02[0-9]{9}\b', # Landline
        ],
        'severity': 'low',  # Phone numbers in comments might be legitimate
        'labels': ['pii', 'phone'],
    },
    # ID numbers
    'id_number': {
        'patterns': [
            r'\b(cmnd|cccd|cmtnd|căn\s*cước)\s*[:.]*\s*\d{9,12}\b',
            r'\b\d{9}\b|\b\d{12}\b', # Standalone ID might be risky, usually relies on context keyword above
        ],
        'severity': 'moderate',
        'labels': ['pii', 'id_number'],
    },
    # NEW: Email addresses
    'email': {
        'patterns': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b(email|mail)\s*[:.]*\s*\S+@\S+\b',
        ],
        'severity': 'low',
        'labels': ['pii', 'email'],
    },
    # NEW: Home Addresses
    'address': {
        'patterns': [
            r'\b(địa\s*chỉ|nhà\s*tại|nhà\s*số|số\s*nhà)\s*[:.]*\s*\d+.*?(phường|quận|huyện|thành\s*phố|tp|tỉnh)\b',
            r'\b\d+\s+(đường|phố|ngõ|ngách|hẻm).*?(phường|quận|huyện|tp)\b',
            # Flexible address
            r'\b(địa\s*chỉ|nhà)\s*(mình|tui|em|tôi|ở)?\s*(là|tại|số)?\s*[:.]*\s*\d+.*?(phường|quận|huyện|tp|tỉnh)\b',
        ],
        'severity': 'moderate',
        'labels': ['pii', 'address'],
    },
}


# ==================== PERSONAL PRONOUNS (targeting indicators) ====================

PERSONAL_ATTACK_INDICATORS = {
    # Second person pronouns (targeting someone)
    'target_pronouns': ['mày', 'mi', 'ngươi', 'bay', 'chúng mày', 'tụi mày', 'bọn mày'],
    
    # Third person (talking about someone)
    'third_person': ['nó', 'thằng này', 'con này', 'đứa này', 'thằng kia', 'con kia'],
    
    # First person (speaker)
    'speaker_pronouns': ['tao', 'tau', 'tui', 'tớ'],
}


# ==================== SAFE WORDS / WHITELIST ====================

# Words that should never trigger detection even if containing toxic substrings
GLOBAL_SAFE_WORDS = {
    # Common Vietnamese words
    'các', 'cách', 'cục', 'lon', 'lòng', 'người', 'những', 'nguồn', 'ngủ',
    'nguyên', 'nguyễn', 'duyên', 'duyệt', 'du lịch', 'du học', 'giáo dục',
    'sử dụng', 'ứng dụng', 'dự án', 'dữ liệu',
    
    # Product review context
    'sản phẩm', 'dịch vụ', 'chất lượng', 'giao hàng', 'đóng gói',
    'shop', 'cửa hàng', 'đánh giá', 'review',
    
    # Edit/Credit/Reddit
    'edit', 'credit', 'reddit', 'editor',
}


# ==================== MAIN CHECKER CLASS ====================

class EnhancedRuleChecker:
    """
    Enhanced rule-based / lexicon checker (Layer B)
    
    Uses multiple text versions from Layer A for comprehensive detection.
    """
    
    def __init__(self):
        # Compile all patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for performance"""
        self.compiled_profanity = {}
        for key, info in PROFANITY_STEMS.items():
            self.compiled_profanity[key] = {
                'patterns': [re.compile(p, re.IGNORECASE | re.UNICODE) for p in info['patterns']],
                'stripped': re.compile(info['stripped_pattern'], re.IGNORECASE) if 'stripped_pattern' in info else None,
                'info': info,
            }
        
        self.compiled_harassment = {}
        for key, info in HARASSMENT_PATTERNS.items():
            self.compiled_harassment[key] = {
                'patterns': [re.compile(p, re.IGNORECASE | re.UNICODE) for p in info['patterns']],
                'info': info,
            }
        
        self.compiled_hate = {}
        for key, info in HATE_SPEECH_PATTERNS.items():
            self.compiled_hate[key] = {
                'patterns': [re.compile(p, re.IGNORECASE | re.UNICODE) for p in info['patterns']],
                'info': info,
            }
            
        self.compiled_spam = {}
        for key, info in SPAM_PATTERNS.items():
            self.compiled_spam[key] = {
                'patterns': [re.compile(p, re.IGNORECASE | re.UNICODE) for p in info['patterns']],
                'info': info,
            }
        
        # NEW: Compile THREAT patterns
        self.compiled_threat = {}
        for key, info in THREAT_PATTERNS.items():
            self.compiled_threat[key] = {
                'patterns': [re.compile(p, re.IGNORECASE | re.UNICODE) for p in info['patterns']],
                'info': info,
            }
        
        # NEW: Compile PII patterns
        self.compiled_pii = {}
        for key, info in PII_PATTERNS.items():
            self.compiled_pii[key] = {
                'patterns': [re.compile(p, re.IGNORECASE | re.UNICODE) for p in info['patterns']],
                'info': info,
            }
            
        # Re-compile expanded profanity patterns if needed
        for key, info in PROFANITY_STEMS.items():
            if 'additional_patterns' in info:
                self.compiled_profanity[key]['patterns'].extend(
                    [re.compile(p, re.IGNORECASE | re.UNICODE) for p in info['additional_patterns']]
                )
    
    def _has_target_pronoun(self, text: str) -> bool:
        """Check if text contains pronouns indicating target (mày/mi/nó...)"""
        text_lower = text.lower()
        
        for pronoun in PERSONAL_ATTACK_INDICATORS['target_pronouns']:
            if pronoun in text_lower:
                return True
        
        for pronoun in PERSONAL_ATTACK_INDICATORS['third_person']:
            if pronoun in text_lower:
                return True
        
        return False
    
    def _is_in_safe_context(self, text: str, word: str, safe_contexts: List[str]) -> bool:
        """Check if word appears in a safe context"""
        text_lower = text.lower()
        
        for context in safe_contexts:
            if context in text_lower:
                return True
        
        return False
    
    def _check_profanity(self, text: str, text_no_diacritics: str) -> List[Dict]:
        """Check for profanity patterns"""
        findings = []
        text_lower = text.lower()
        
        for key, compiled in self.compiled_profanity.items():
            info = compiled['info']
            
            # Check safe contexts
            safe_contexts = info.get('safe_contexts', [])
            if safe_contexts and self._is_in_safe_context(text, key, safe_contexts):
                continue
            
            # Check if context required (like "ngu" needs full pattern)
            if info.get('context_required'):
                # Only match full patterns, not standalone
                for pattern in compiled['patterns']:
                    match = pattern.search(text_lower)
                    if match:
                        findings.append({
                            'type': 'profanity',
                            'key': key,
                            'matched': match.group(),
                            'severity': info['severity'],
                            'labels': info['labels'],
                        })
                        break
            else:
                # Check main patterns
                for pattern in compiled['patterns']:
                    match = pattern.search(text_lower)
                    if match:
                        findings.append({
                            'type': 'profanity',
                            'key': key,
                            'matched': match.group(),
                            'severity': info['severity'],
                            'labels': info['labels'],
                        })
                        break
                
                # Also check stripped pattern on no-diacritics version
                if not findings or findings[-1]['key'] != key:
                    if compiled['stripped']:
                        match = compiled['stripped'].search(text_no_diacritics)
                        if match:
                            # Double-check not in safe context
                            if not self._is_in_safe_context(text, key, safe_contexts):
                                findings.append({
                                    'type': 'profanity',
                                    'key': key,
                                    'matched': match.group(),
                                    'severity': info['severity'],
                                    'labels': info['labels'],
                                    'from_stripped': True,
                                })
        
        return findings
    
    def _check_harassment(self, text: str) -> List[Dict]:
        """Check for harassment/body-shaming patterns"""
        findings = []
        text_lower = text.lower()
        
        for key, compiled in self.compiled_harassment.items():
            info = compiled['info']
            
            # Check if requires target
            if info.get('requires_target') and not self._has_target_pronoun(text):
                continue
            
            for pattern in compiled['patterns']:
                match = pattern.search(text_lower)
                if match:
                    findings.append({
                        'type': 'harassment',
                        'key': key,
                        'matched': match.group(),
                        'severity': info['severity'],
                        'labels': info['labels'],
                    })
                    break
        
        return findings
    
    def _check_hate_speech(self, text: str) -> List[Dict]:
        """Check for hate speech patterns"""
        findings = []
        text_lower = text.lower()
        
        for key, compiled in self.compiled_hate.items():
            info = compiled['info']
            
            # Check additional context requirement
            additional_context = info.get('additional_context', [])
            if additional_context:
                has_context = any(ctx in text_lower for ctx in additional_context)
                if not has_context:
                    continue
            
            for pattern in compiled['patterns']:
                match = pattern.search(text_lower)
                if match:
                    findings.append({
                        'type': 'hate_speech',
                        'key': key,
                        'matched': match.group(),
                        'severity': info['severity'],
                        'labels': info['labels'],
                    })
                    break
        
        return findings

    def _check_spam(self, text: str) -> List[Dict]:
        """Check for spam patterns"""
        findings = []
        text_lower = text.lower()
        
        for key, compiled in self.compiled_spam.items():
            info = compiled['info']
            for pattern in compiled['patterns']:
                match = pattern.search(text_lower)
                if match:
                    findings.append({
                        'type': 'spam',
                        'key': key,
                        'matched': match.group(),
                        'severity': info['severity'],
                        'labels': info['labels'],
                    })
                    break
        return findings
    
    def _check_threat(self, text: str) -> List[Dict]:
        """Check for threat/violence patterns"""
        findings = []
        text_lower = text.lower()
        
        for key, compiled in self.compiled_threat.items():
            info = compiled['info']
            for pattern in compiled['patterns']:
                match = pattern.search(text_lower)
                if match:
                    findings.append({
                        'type': 'threat',
                        'key': key,
                        'matched': match.group(),
                        'severity': info['severity'],
                        'labels': info['labels'],
                    })
                    break
        return findings
    
    def _check_pii(self, text: str) -> List[Dict]:
        """Check for PII (Personal Identifiable Information) patterns"""
        findings = []
        text_lower = text.lower()
        
        for key, compiled in self.compiled_pii.items():
            info = compiled['info']
            for pattern in compiled['patterns']:
                match = pattern.search(text_lower)
                if match:
                    findings.append({
                        'type': 'pii',
                        'key': key,
                        'matched': match.group(),
                        'severity': info['severity'],
                        'labels': info['labels'],
                    })
                    break
        return findings
    
    def check(
        self, 
        text: str, 
        normalized_text: str = None, 
        no_diacritics_text: str = None,
        metadata: Dict = None
    ) -> Optional[Dict[str, Any]]:
        """
        Main check method.
        
        Args:
            text: Original text
            normalized_text: Fully normalized text from Layer A
            no_diacritics_text: Text with Vietnamese diacritics removed
            metadata: Normalization metadata from Layer A
        
        Returns:
            Result dict if violation found, None if clean
        """
        # Use original if normalized not provided
        if normalized_text is None:
            normalized_text = text.lower()
        if no_diacritics_text is None:
            no_diacritics_text = text.lower()
        
        all_findings = []
        
        # Check all categories
        profanity = self._check_profanity(normalized_text, no_diacritics_text)
        all_findings.extend(profanity)
        
        harassment = self._check_harassment(text)  # Use original for pronoun checking
        all_findings.extend(harassment)
        
        hate = self._check_hate_speech(text)  # Use original for full context
        all_findings.extend(hate)

        spam = self._check_spam(text)
        all_findings.extend(spam)
        
        # NEW: Check for threats/violence
        threat = self._check_threat(text)
        all_findings.extend(threat)
        
        # NEW: Check for PII
        pii = self._check_pii(text)
        all_findings.extend(pii)
        
        # Special check: obfuscated insults
        # If obfuscation was detected and normalized text contains insult words,
        # this indicates intentional bypass attempt
        if metadata and metadata.get('has_obfuscation'):
            obfuscated_insults_info = PROFANITY_STEMS.get('obfuscated_insults', {})
            standalone_words = obfuscated_insults_info.get('standalone_words', [])
            
            for word in standalone_words:
                # Check if normalized text contains this word as standalone
                if re.search(rf'\b{word}\b', normalized_text, re.IGNORECASE):
                    # Check if original text didn't contain it (meaning it was obfuscated)
                    if not re.search(rf'\b{word}\b', text.lower(), re.IGNORECASE):
                        all_findings.append({
                            'type': 'obfuscated_insult',
                            'key': 'obfuscated_insults',
                            'matched': word,
                            'severity': 'moderate',
                            'labels': ['insult', 'obfuscation_bypass'],
                        })
                        break
        
        if not all_findings:
            return None
        
        # Determine overall severity and action
        has_severe = any(f['severity'] == 'severe' for f in all_findings)
        has_hate = any(f['type'] == 'hate_speech' for f in all_findings)
        has_spam = any(f['type'] == 'spam' for f in all_findings)
        has_harassment = any(f['type'] == 'harassment' for f in all_findings)
        has_body_shaming = 'body_shaming' in [l for f in all_findings for l in f.get('labels', [])]
        has_threat = any(f['type'] == 'threat' for f in all_findings)
        has_pii = any(f['type'] == 'pii' for f in all_findings)
        
        # NEW: Escalation logic for body-shaming
        # Escalate to reject if severe expressions are used
        escalate_body_shaming = False
        if has_body_shaming or has_harassment:
            text_lower = text.lower() if 'text' in dir() else normalized_text
            severe_expressions = [
                'muốn nôn', 'ghê tởm', 'kinh tởm', 'kinh khủng', 'ghê ghớm',
                'đáng chết', 'chết đi', 'biến đi', 'cút đi',
                'xấu kinh', 'xấu ghê', 'xấu tởm', 'xấu khủng',
                'béo như lợn', 'gầy như que', 'đen như than',
                'mặt như l*', 'mặt l*', 'mặt như đít',
            ]
            for expr in severe_expressions:
                if expr in text_lower:
                    escalate_body_shaming = True
                    break
        
        # Collect all labels
        all_labels = set()
        for f in all_findings:
            all_labels.update(f['labels'])
        
        # NEW: Escalation for any personal attack containing "ngu"
        has_ngu_attack = any(
            'ngu' in f.get('matched', '').lower()
            and f['type'] in ('profanity', 'harassment', 'obfuscated_insult')
            for f in all_findings
        )
        
        # Determine action - THREATS and 'ngu' attacks are always REJECT
        if has_hate or has_severe or escalate_body_shaming or has_spam or has_threat or has_ngu_attack:
            action = 'reject'
            confidence = 0.95
        elif has_pii:
            action = 'review'  # PII should be reviewed, may be legitimate
            confidence = 0.75
        else:
            action = 'review'
            confidence = 0.80
        
        # Build reasoning
        matched_items = [f['matched'] for f in all_findings[:3]]
        types = set(f['type'] for f in all_findings)
        
        reasoning_parts = []
        if 'hate_speech' in types:
            reasoning_parts.append(' HATE SPEECH')
        if 'threat' in types:
            reasoning_parts.append(' THREAT/VIOLENCE')
        if 'harassment' in types or 'obfuscated_insult' in types:
            if escalate_body_shaming:
                reasoning_parts.append(' SEVERE HARASSMENT')
            else:
                reasoning_parts.append(' HARASSMENT')
        if 'profanity' in types:
            reasoning_parts.append(' PROFANITY')
        if 'spam' in types:
            reasoning_parts.append(' SPAM')
        if 'pii' in types:
            reasoning_parts.append(' PII DETECTED')
        
        reasoning = f"{', '.join(reasoning_parts)}: {', '.join(matched_items)}"
        
        # Add obfuscation note if detected
        if metadata and metadata.get('has_obfuscation'):
            reasoning += f" (obfuscation: {', '.join(metadata['obfuscation_types'])})"
        
        return {
            'action': action,
            'labels': list(all_labels),
            'confidence': confidence,
            'reasoning': reasoning,
            'findings': all_findings,
            'method': 'rule_based_enhanced',
            'has_obfuscation': metadata.get('has_obfuscation', False) if metadata else False,
            'escalated': escalate_body_shaming,
        }


# ==================== SINGLETON INSTANCE ====================

_checker_instance = None

def get_rule_checker() -> EnhancedRuleChecker:
    """Get singleton checker instance"""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = EnhancedRuleChecker()
    return _checker_instance


# ==================== TEST ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Import normalizer
    try:
        from text_normalizer import get_normalizer
    except ImportError:
        from nlp.text_normalizer import get_normalizer
    
    normalizer = get_normalizer()
    checker = get_rule_checker()
    
    test_cases = [
        # Profanity
        "đm mày",
        "vcl",
        "dm con chó",
        
        # Obfuscated profanity
        "d.m",
        "đ.m",
        "n.g.u",
        "d:m",
        "d:m,m",
        
        # Harassment / body-shaming (key test cases from screenshots)
        "Sao mày xấu thế, nhìn mặt mày tao muốn nôn",
        "đồ ngu ngốc",
        "thằng này ngu quá",
        
        # Hate speech (key test case from screenshot)
        "Bọn da đen bẩn thỉu cút về nước đi",
        "tàu khựa",
        
        # NEW: Spam/Scam test cases (reported issues)
        "Bạn trúng 100 triệu! Click để nhận: http://scam.vn",  # Prize scam
        "Nhắn tin cho anh, anh cho xem cái hay",  # Solicitation
        "Like và share để nhiều người biết đến hơn!",  # Engagement bait
        "inbox em để nhận link hot",  # Solicitation
        "Chúc mừng bạn trúng thưởng, click link để nhận",  # Prize scam
        
        # NEW: Threat/Violence test cases (CRITICAL - reported issues)
        "Tao sẽ tìm mày và đánh cho chừa",  # Physical threat - MUST REJECT
        "Mày sẽ chết",  # Death threat
        "Tao biết nhà mày, liệu hồn",  # Intimidation/stalking
        "Đánh cho chừa thói",  # Violence
        
        # NEW: PII test cases
        "STK: Vietcombank 1234567890",  # Bank account
        "Liên hệ zalo 0912345678 để mua",  # Phone + spam
        
        # Safe content
        "Sản phẩm tốt quá",
        "Lon bia này ngon",
        "Các bạn có khỏe không?",
        "Hài lòng với dịch vụ",
        
        # Edge cases - SHOULD remain ALLOWED (valid feedback)
        "Sản phẩm tệ quá, thất vọng",  # Negative but valid feedback
        "Tôi không hài lòng với dịch vụ",  # Valid complaint
        "Video hay nhưng hơi dài, nên tóm tắt lại",  # Constructive feedback
        "Góc nhìn mới lạ, nhưng thiếu phân tích sâu",  # Balanced critique
        "Bài viết rất hay và bổ ích! Cảm ơn tác giả đã chia sẻ 👍",  # Positive - no violation
        "Mình đồng ý 100% với quan điểm này",  # Positive agreement
    ]
    
    print("=" * 80)
    print("ENHANCED RULE CHECKER TEST")
    print("=" * 80)
    
    for text in test_cases:
        print(f"\n📝 Input: '{text}'")
        
        # Get normalized versions
        versions = normalizer.create_all_versions(text)
        
        # Run checker
        result = checker.check(
            text=text,
            normalized_text=versions['fully_normalized'],
            no_diacritics_text=versions['no_diacritics'],
            metadata=versions['metadata']
        )
        
        if result:
            print(f"    VIOLATION: {result['reasoning']}")
            print(f"   Action: {result['action']}, Labels: {result['labels']}")
            print(f"   Confidence: {result['confidence']:.2%}")
        else:
            print(f"    CLEAN")
        
        print("-" * 60)
