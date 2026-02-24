import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Tuple
import logging
import os
import re

from nlp.preprocessing import preprocess_vietnamese_text, is_text_valid
from nlp.toxic_words import (
    SEVERE_PROFANITY,
    SEVERE_INSULTS,
    HATE_LGBTQ,
    HATE_RACISM,
    HATE_RELIGION,
    HATE_SEXISM,
    SEXUAL_EXPLICIT,
    SEXUAL_SUGGESTIVE,
    SEXUAL_SOLICITATION,
    MODERATE_NEGATIVE,
    PERSONAL_ATTACKS,
    SPAM_INDICATORS,
    TOXIC_PATTERNS,
    SEVERITY_BOOSTERS,
    SEVERITY_REDUCERS,
    ALLOWED_PHRASES,
    OPINION_CRITICISM_CONTEXT,
    SEVERITY_SCORES,
    REJECT_THRESHOLD,
    REVIEW_THRESHOLD,
    WARNING_THRESHOLD,
    AUTO_REJECT_CATEGORIES,
    CONTEXT_MULTIPLIERS,
    get_all_toxic_words,
    get_critical_words,
    get_hate_speech_words,
    get_sexual_content_words,
    is_auto_reject_category,
)
from nlp.sentiment_words import (
    HIGHLY_POSITIVE,
    MODERATELY_POSITIVE,
    SLIGHTLY_POSITIVE,
    HIGHLY_NEGATIVE,
    MODERATELY_NEGATIVE,
    SLIGHTLY_NEGATIVE,
    NEUTRAL_WORDS,
    POSITIVE_PHRASES,
    NEGATIVE_PHRASES,
    POSITIVE_EMOJIS,
    NEGATIVE_EMOJIS,
    NEUTRAL_EMOJIS,
    SENTIMENT_SCORES,
    EMOJI_SCORE,
    PHRASE_SCORE,
    POSITIVE_THRESHOLD,
    NEGATIVE_THRESHOLD,
    INTENSIFIERS,
    NEGATIONS,
)

logger = logging.getLogger(__name__)


class ModerationInference:
    """NLP inference for Vietnamese text moderation"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model_path = model_path or 'vinai/phobert-base-v2'
        self.device = device
        self.tokenizer = None
        self.model = None
        
        # UIT-NLP Model Labels mapping
        self.id2label = {0: 'clean', 1: 'offensive', 2: 'hate'}
        self.label2id = {'clean': 0, 'offensive': 1, 'hate': 2}
        
        self.load_model()
    
    def analyze_sentiment_rule_based(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using rule-based approach
        Excellent for short texts where ML models struggle
        
        Returns:
            Dict with sentiment, score, confidence, matched_words
        """
        text_lower = text.lower()
        normalized_text = preprocess_vietnamese_text(text_lower)
        
        sentiment_score = 0
        matched_positive = []
        matched_negative = []
        matched_neutral = []
        has_negation = False
        intensifier_multiplier = 1.0
        
        # Check for negations
        for negation in NEGATIONS:
            if negation in normalized_text:
                has_negation = True
                break
        
        # Check for intensifiers
        for intensifier, multiplier in INTENSIFIERS.items():
            if intensifier in normalized_text:
                intensifier_multiplier = max(intensifier_multiplier, multiplier)
        
        # Check emojis first (strong signals)
        for emoji in POSITIVE_EMOJIS:
            if emoji in text:
                sentiment_score += EMOJI_SCORE['POSITIVE']
                matched_positive.append(emoji)
        
        for emoji in NEGATIVE_EMOJIS:
            if emoji in text:
                sentiment_score += EMOJI_SCORE['NEGATIVE']
                matched_negative.append(emoji)
        
        # Check positive phrases (multi-word expressions)
        for phrase in POSITIVE_PHRASES:
            if phrase in normalized_text:
                sentiment_score += PHRASE_SCORE['POSITIVE']
                matched_positive.append(phrase)
        
        # Check negative phrases
        for phrase in NEGATIVE_PHRASES:
            if phrase in normalized_text:
                sentiment_score += PHRASE_SCORE['NEGATIVE']
                matched_negative.append(phrase)
        
        # Check highly positive words
        for word in HIGHLY_POSITIVE:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                sentiment_score += SENTIMENT_SCORES['HIGHLY_POSITIVE']
                matched_positive.append(word)
        
        # Check moderately positive words
        for word in MODERATELY_POSITIVE:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                sentiment_score += SENTIMENT_SCORES['MODERATELY_POSITIVE']
                matched_positive.append(word)
        
        # Check slightly positive words
        for word in SLIGHTLY_POSITIVE:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                sentiment_score += SENTIMENT_SCORES['SLIGHTLY_POSITIVE']
                matched_positive.append(word)
        
        # Check highly negative words (non-toxic)
        for word in HIGHLY_NEGATIVE:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                sentiment_score += SENTIMENT_SCORES['HIGHLY_NEGATIVE']
                matched_negative.append(word)
        
        # Check moderately negative words
        for word in MODERATELY_NEGATIVE:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                sentiment_score += SENTIMENT_SCORES['MODERATELY_NEGATIVE']
                matched_negative.append(word)
        
        # Check slightly negative words
        for word in SLIGHTLY_NEGATIVE:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                sentiment_score += SENTIMENT_SCORES['SLIGHTLY_NEGATIVE']
                matched_negative.append(word)
        
        # Check neutral words
        for word in NEUTRAL_WORDS:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                matched_neutral.append(word)
        
        # Apply intensifier
        if intensifier_multiplier > 1.0 and sentiment_score != 0:
            sentiment_score = int(sentiment_score * intensifier_multiplier)
        
        # Apply negation (reverse sentiment)
        if has_negation and sentiment_score != 0:
            sentiment_score = -sentiment_score
        
        # Determine sentiment
        if sentiment_score >= POSITIVE_THRESHOLD:
            sentiment = 'positive'
            confidence = min(0.95, 0.7 + abs(sentiment_score) / 50)
        elif sentiment_score <= NEGATIVE_THRESHOLD:
            sentiment = 'negative'
            confidence = min(0.95, 0.7 + abs(sentiment_score) / 50)
        else:
            sentiment = 'neutral'
            confidence = 0.6
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'confidence': round(confidence, 4),
            'matched_positive': list(set(matched_positive)),
            'matched_negative': list(set(matched_negative)),
            'matched_neutral': list(set(matched_neutral)),
            'has_negation': has_negation,
            'intensifier': intensifier_multiplier
        }
    
    def detect_toxic_words(self, text: str) -> Tuple[int, List[str], Dict[str, Any]]:
        """
        Detect toxic/profanity words with ENTERPRISE-LEVEL detection
        
        NEW FEATURES:
        - Hate speech detection (LGBTQ+, racism, religion, sexism)
        - Sexual content detection (explicit, suggestive, solicitation)
        - Context-aware scoring with multipliers
        - Auto-reject categories
        - Enhanced pattern matching
        
        Returns:
            (severity_score, flagged_words, details)
        """
        text_lower = text.lower()
        normalized_text = preprocess_vietnamese_text(text_lower)
        
        # Collect allowed words found in text (for skipping in pattern matching)
        # Don't return early - just track which words should be excluded
        allowed_words_in_text = set()
        for phrase in ALLOWED_PHRASES:
            if phrase in normalized_text:
                allowed_words_in_text.add(phrase)
                # Also add individual words from phrase
                for word in phrase.split():
                    allowed_words_in_text.add(word)
        
        # ===== CONTEXT DETECTION =====
        context_flags = {
            'is_opinion_criticism': False,
            'is_product_review': False,
            'has_personal_pronoun': False,
            'has_severity_booster': False,
            'has_negation': False,
            'targeting_group': False,
        }
        
        # Check if criticizing opinion/idea (not person)
        for context_word in OPINION_CRITICISM_CONTEXT:
            if context_word in normalized_text:
                context_flags['is_opinion_criticism'] = True
                break
        
        # Check for product review context
        product_review_keywords = ['sản phẩm', 'san pham', 'dịch vụ', 'dich vu', 'shop', 'cửa hàng', 'cua hang']
        for keyword in product_review_keywords:
            if keyword in normalized_text:
                context_flags['is_product_review'] = True
                break
        
        # Check for personal pronouns (attack on person)
        personal_pronouns = ['mày', 'mi', 'tao', 'tau', 'm', 't']
        for pronoun in personal_pronouns:
            if re.search(r'\b' + re.escape(pronoun) + r'\b', normalized_text):
                context_flags['has_personal_pronoun'] = True
                break
        
        # Check for severity boosters
        for booster in SEVERITY_BOOSTERS:
            if booster in normalized_text:
                context_flags['has_severity_booster'] = True
                break
        
        # Check for negation
        for negation in SEVERITY_REDUCERS:
            if negation in normalized_text:
                context_flags['has_negation'] = True
                break
        
        # ===== WORD DETECTION =====
        flagged_words = []
        severity_score = 0
        auto_reject_reason = None
        category_hits = {
            'severe_profanity': [],
            'severe_insults': [],
            'hate_lgbtq': [],
            'hate_racism': [],
            'hate_religion': [],
            'hate_sexism': [],
            'sexual_explicit': [],
            'sexual_suggestive': [],
            'sexual_solicitation': [],
            'moderate_negative': [],
            'personal_attacks': [],
            'spam': [],
            'patterns': []
        }
        
        # ===== 1. HATE SPEECH DETECTION (HIGHEST PRIORITY) =====
        # LGBTQ+ hate
        for word in HATE_LGBTQ:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                flagged_words.append(word)
                category_hits['hate_lgbtq'].append(word)
                severity_score += SEVERITY_SCORES['HATE_LGBTQ']
                context_flags['targeting_group'] = True
                auto_reject_reason = f"Hate speech: LGBTQ+ discrimination - {word}"
        
        # Racism
        for word in HATE_RACISM:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                flagged_words.append(word)
                category_hits['hate_racism'].append(word)
                severity_score += SEVERITY_SCORES['HATE_RACISM']
                context_flags['targeting_group'] = True
                auto_reject_reason = f"Hate speech: Racial discrimination - {word}"
        
        # Religious hate
        for word in HATE_RELIGION:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                flagged_words.append(word)
                category_hits['hate_religion'].append(word)
                severity_score += SEVERITY_SCORES['HATE_RELIGION']
                context_flags['targeting_group'] = True
                auto_reject_reason = f"Hate speech: Religious discrimination - {word}"
        
        # Sexism
        for word in HATE_SEXISM:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                flagged_words.append(word)
                category_hits['hate_sexism'].append(word)
                severity_score += SEVERITY_SCORES['HATE_SEXISM']
                context_flags['targeting_group'] = True
                auto_reject_reason = f"Hate speech: Gender discrimination - {word}"
        
        # ===== 2. SEXUAL CONTENT DETECTION =====
        # Explicit sexual content
        for word in SEXUAL_EXPLICIT:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                flagged_words.append(word)
                category_hits['sexual_explicit'].append(word)
                severity_score += SEVERITY_SCORES['SEXUAL_EXPLICIT']
                auto_reject_reason = f"Explicit pornographic content - {word}"
        
        # Suggestive sexual content
        for word in SEXUAL_SUGGESTIVE:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                flagged_words.append(word)
                category_hits['sexual_suggestive'].append(word)
                severity_score += SEVERITY_SCORES['SEXUAL_SUGGESTIVE']
        
        # Sexual solicitation
        for word in SEXUAL_SOLICITATION:
            if word in normalized_text:  # Không cần word boundary cho phrases
                flagged_words.append(word)
                category_hits['sexual_solicitation'].append(word)
                severity_score += SEVERITY_SCORES['SEXUAL_SOLICITATION']
                auto_reject_reason = f"Sexual solicitation - {word}"
        
        # ===== 3. PROFANITY & INSULTS =====
        # Severe profanity
        for word in SEVERE_PROFANITY:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                flagged_words.append(word)
                category_hits['severe_profanity'].append(word)
                severity_score += SEVERITY_SCORES['SEVERE_PROFANITY']
        
        # Severe insults (với context awareness)
        intelligence_insults = {'ngu', 'ngu ngốc', 'ngu người', 'đần', 'đần độn', 'ngớ ngẩn', 'stupid', 'idiot', 'moron', 'dumb'}
        
        # CRITICAL: Vietnamese words that CONTAIN "ngu" but are NOT insults
        # These are common words/names where \b pattern incorrectly matches
        ngu_safe_words = {
            'nguyễn', 'nguyen', 'nguyên', 'nguyển', 'nguyện', 'nguyệt',
            'người', 'những', 'nguồn', 'ngủ', 'ngũ', 'nguội', 'ngước',
            'ngựa', 'ngứa', 'ngụ', 'ngư', 'nguy', 'nguội',
            'nguyễn thị', 'nguyễn văn', 'nguyễn ngọc', 'nguyễn minh', 'nguyễn hữu',
        }
        
        for word in SEVERE_INSULTS:
            pattern = r'\b' + re.escape(word) + r'\b'
            match = re.search(pattern, normalized_text, re.IGNORECASE)
            if match:
                # CRITICAL FIX: Check if "ngu" match is actually part of a larger word
                # Vietnamese diacritics can break word boundaries
                if word == 'ngu':
                    # Check if any safe word containing "ngu" is in the text
                    is_part_of_safe_word = False
                    for safe_word in ngu_safe_words:
                        if safe_word in normalized_text:
                            is_part_of_safe_word = True
                            break
                    
                    # Also check the actual text around the match
                    start, end = match.span()
                    # Check if there are more Vietnamese letters after "ngu"
                    if end < len(normalized_text):
                        next_char = normalized_text[end]
                        # Vietnamese letters that commonly follow "ngu" in legitimate words
                        if next_char in 'yễơờởỡợủũụưừứửữựiịìíỉĩôồốộổỗơ':
                            is_part_of_safe_word = True
                    
                    if is_part_of_safe_word:
                        continue  # Skip - this is part of a legitimate word
                
                # Allow opinion criticism (e.g., "quan điểm ngu si")
                if context_flags['is_opinion_criticism'] and word in intelligence_insults:
                    continue  # Skip - allowed
                
                flagged_words.append(word)
                category_hits['severe_insults'].append(word)
                severity_score += SEVERITY_SCORES['SEVERE_INSULTS']
        
        # ===== 4. MODERATE VIOLATIONS =====
        # Moderate negative
        for word in MODERATE_NEGATIVE:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                flagged_words.append(word)
                category_hits['moderate_negative'].append(word)
                severity_score += SEVERITY_SCORES['MODERATE_NEGATIVE']
        
        # Personal attacks - ONLY add to score if there are other violations
        # "mày, tao" alone is NOT a violation - it's casual Vietnamese speech
        personal_attack_words_found = []
        for word in PERSONAL_ATTACKS:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                personal_attack_words_found.append(word)
                category_hits['personal_attacks'].append(word)
        
        # Only count PERSONAL_ATTACKS if there are OTHER violations (profanity, insults, etc)
        has_other_violations = (
            category_hits['severe_profanity'] or 
            category_hits['severe_insults'] or
            category_hits['hate_lgbtq'] or
            category_hits['hate_racism'] or
            category_hits['hate_religion'] or
            category_hits['hate_sexism'] or
            category_hits['sexual_explicit'] or
            category_hits['sexual_suggestive'] or
            category_hits['sexual_solicitation'] or
            category_hits['moderate_negative']
        )
        
        if personal_attack_words_found and has_other_violations:
            # Add to flagged words and score ONLY if combined with real violations
            flagged_words.extend(personal_attack_words_found)
            severity_score += SEVERITY_SCORES['PERSONAL_ATTACKS'] * len(personal_attack_words_found)
        
        # Spam indicators
        for word in SPAM_INDICATORS:
            if word in normalized_text:
                flagged_words.append(word)
                category_hits['spam'].append(word)
                severity_score += SEVERITY_SCORES['SPAM_INDICATORS']
        
        # ===== 5. PATTERN MATCHING =====
        for pattern in TOXIC_PATTERNS:
            matches = re.findall(pattern, normalized_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    # CRITICAL: Skip if match is an allowed word (e.g., "các" matched by "cặc" pattern)
                    match_lower = match.lower() if isinstance(match, str) else str(match).lower()
                    if match_lower in allowed_words_in_text:
                        logger.debug(f"Skipping allowed word in pattern match: {match}")
                        continue
                    # Also skip common Vietnamese words that are false positives
                    common_words = {
                        'các', 'cách', 'cục', 'cắc', 'cạc',  # Similar to cặc
                        'người', 'nguồn', 'nguyên', 'ngủ',   # Similar to ngu
                        'dùng', 'dũng', 'dũ',                 # Similar to đụ
                        'lòng', 'lồng', 'long',               # Similar to lồn
                        'đột',                                # Random
                        'đề', 'để', 'đe', 'dề', 'dể', 'de',   # Similar to đéo
                        'deo', 'đeo'                          # đeo (wear) is NOT đéo
                    }
                    if match_lower in common_words:
                        logger.debug(f"Skipping common Vietnamese word: {match}")
                        continue
                    flagged_words.append(match)
                    category_hits['patterns'].append(match)
                severity_score += SEVERITY_SCORES['TOXIC_PATTERNS'] * len([m for m in matches if m.lower() not in allowed_words_in_text and m.lower() not in common_words])
        
        # ===== APPLY CONTEXT MULTIPLIERS =====
        original_score = severity_score
        
        if context_flags['has_personal_pronoun']:
            severity_score = int(severity_score * CONTEXT_MULTIPLIERS['has_personal_pronoun'])
        
        if context_flags['has_severity_booster']:
            severity_score = int(severity_score * CONTEXT_MULTIPLIERS['has_severity_booster'])
        
        if context_flags['targeting_group']:
            severity_score = int(severity_score * CONTEXT_MULTIPLIERS['targeting_group'])
        
        # Count violations
        violation_count = sum(1 for hits in category_hits.values() if hits)
        if violation_count >= 2:
            severity_score = int(severity_score * CONTEXT_MULTIPLIERS['multiple_violations'])
        
        # Reduce for legitimate contexts
        if context_flags['is_opinion_criticism'] and not context_flags['has_personal_pronoun']:
            severity_score = int(severity_score * CONTEXT_MULTIPLIERS['opinion_criticism'])
        
        if context_flags['is_product_review'] and not context_flags['has_personal_pronoun']:
            severity_score = int(severity_score * CONTEXT_MULTIPLIERS['product_review'])
        
        if context_flags['has_negation']:
            severity_score = int(severity_score * CONTEXT_MULTIPLIERS['has_negation'])
        
        # Remove duplicates
        flagged_words = list(set(flagged_words))
        
        details = {
            'category_hits': category_hits,
            'score': severity_score,
            'original_score': original_score,
            'flagged_count': len(flagged_words),
            'context_flags': context_flags,
            'auto_reject_reason': auto_reject_reason,
            'violation_count': violation_count
        }
        
        return severity_score, flagged_words, details
    
    def load_model(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Ưu tiên load model "chính chủ" đã fine-tune (nếu có)
            custom_model_path = '/app/models/custom_phobert'
            
            if os.path.exists(custom_model_path):
                logger.info(f"Loading CUSTOM FINE-TUNED model from {custom_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    custom_model_path,
                    num_labels=3
                )
            elif os.path.exists(self.model_path):
                logger.info("Loading from local path")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path
                    # AutoConfig will usually handle num_labels for fine-tuned models
                )
            else:
                # Download from HuggingFace
                # Fallback về model gốc ổn định nhất
                logger.info(f"Downloading model {self.model_path} from HuggingFace")
                self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    'vinai/phobert-base-v2',
                    num_labels=3
                )
                
                # Save for future use
                if self.model_path != 'vinai/phobert-base-v2':
                    os.makedirs(self.model_path, exist_ok=True)
                    self.tokenizer.save_pretrained(self.model_path)
                    self.model.save_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment and moderation result
        
        Args:
            text: Input Vietnamese text
        
        Returns:
            Dict with sentiment, moderation_result, confidence, reasoning
        """
        # CRITICAL: First check for toxic words (rule-based - HIGHEST PRIORITY)
        # This MUST happen before sentiment analysis to catch sexual/hate content
        # that might contain positive words like "xinh", "đẹp"
        severity_score, flagged_words, toxicity_details = self.detect_toxic_words(text)
        
        # Determine action based on severity score
        if severity_score >= REJECT_THRESHOLD:
            # Auto reject - severe toxic content
            top_flags = flagged_words[:3] if len(flagged_words) <= 3 else flagged_words[:2] + [f'and {len(flagged_words)-2} other words']
            return {
                'sentiment': 'negative',
                'moderation_result': 'reject',
                'confidence': min(0.95, 0.75 + (severity_score / 100)),
                'reasoning': f'Severe violation - Detected words: {", ".join(top_flags)} (Score: {severity_score})',
                'flagged_words': flagged_words,
                'toxicity_score': severity_score
            }
        elif severity_score >= REVIEW_THRESHOLD:
            # Needs review - moderate toxic content
            return {
                'sentiment': 'negative',
                'moderation_result': 'review',
                'confidence': 0.7,
                'reasoning': f'Content needs review - Detected: {", ".join(flagged_words[:2])} (Score: {severity_score})',
                'flagged_words': flagged_words,
                'toxicity_score': severity_score
            }
        
        # Validate text
        is_valid, reason = is_text_valid(text)
        if not is_valid:
            # Only reject if completely invalid (empty, spam, etc)
            # Don't reject for short text anymore
            if 'ngắn' not in reason:
                return {
                    'sentiment': 'neutral',
                    'moderation_result': 'reject',
                    'confidence': 1.0,
                    'reasoning': reason
                }
        
        # Check text length to determine analysis method
        word_count = len(text.split())
        
        # For short texts (1-3 words), use rule-based sentiment analysis
        # ML models struggle with very short texts
        if word_count <= 3:
            sentiment_result = self.analyze_sentiment_rule_based(text)
            
            # If no sentiment detected, default to neutral
            if sentiment_result['score'] == 0 and not sentiment_result['matched_positive'] and not sentiment_result['matched_negative']:
                return {
                    'sentiment': 'neutral',
                    'moderation_result': 'allowed',
                    'confidence': 0.75,
                    'reasoning': 'Short text, no clear sentiment detected'
                }
            
            # Determine moderation based on sentiment
            sentiment = sentiment_result['sentiment']
            confidence = sentiment_result['confidence']
            
            if sentiment == 'positive':
                moderation = 'allowed'
                matched = ', '.join(sentiment_result['matched_positive'][:3])
                reasoning = f'Positive - Detected: {matched}' if matched else 'Positive'
            elif sentiment == 'negative':
                # Allow legitimate negative reviews/feedback
                # Only flag if extremely toxic (score <= -20) or contains personal attacks
                moderation = 'allowed'
                matched = ', '.join(sentiment_result['matched_negative'][:3])
                reasoning = f'Negative feedback - {matched}' if matched else 'Negative review, acceptable'
            else:
                moderation = 'allowed'
                reasoning = 'Neutral'
            
            return {
                'sentiment': sentiment,
                'moderation_result': moderation,
                'confidence': confidence,
                'reasoning': reasoning,
                'flagged_words': [],
                'sentiment_score': sentiment_result['score']
            }
        
        # For longer texts (4+ words), use hybrid approach:
        # 1. Try rule-based first for better accuracy
        # 2. Fall back to ML model if rule-based is inconclusive
        
        sentiment_result = self.analyze_sentiment_rule_based(text)
        
        # If rule-based found clear sentiment (score != 0), use it
        if abs(sentiment_result['score']) >= 5:
            sentiment = sentiment_result['sentiment']
            confidence = sentiment_result['confidence']
            
            # Determine moderation
            moderation = self._determine_moderation(sentiment, confidence)
            
            if sentiment == 'positive':
                matched = ', '.join(sentiment_result['matched_positive'][:3])
                reasoning = f'Positive - {matched}' if matched else 'Positive content'
            elif sentiment == 'negative':
                matched = ', '.join(sentiment_result['matched_negative'][:3])
                reasoning = f'Negative - {matched}' if matched else 'Negative content'
            else:
                reasoning = 'Neutral content'
            
            return {
                'sentiment': sentiment,
                'moderation_result': moderation,
                'confidence': confidence,
                'reasoning': reasoning,
                'flagged_words': [],
                'sentiment_score': sentiment_result['score']
            }
        
        # Fall back to ML model (PhoBERT Fine-tuned) for ambiguous cases
        # Preprocess
        preprocessed_text = preprocess_vietnamese_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            preprocessed_text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)
        
        # Get results
        predicted_idx = predicted.item()
        predicted_label = self.id2label.get(predicted_idx, 'clean') # 0: clean, 1: offensive, 2: hate
        confidence_score = confidence.item()
        
        # === INTEGRATION LOGIC: Map Hate Speech Labels to Moderation Actions ===
        
        # Label 0: Clean -> Allowed
        # Label 1: Offensive -> Review (or Reject if high confidence)
        # Label 2: Hate -> Reject
        
        moderation_result = 'allowed'
        reasoning = 'Content is safe'
        
        if predicted_label == 'hate':
            moderation_result = 'reject'
            reasoning = f'Hate speech detected by AI ({confidence_score:.2%} confidence)'
        
        elif predicted_label == 'offensive':
            if confidence_score > 0.85:
                 moderation_result = 'reject'
                 reasoning = f'Highly offensive content detected ({confidence_score:.2%} confidence)'
            else:
                 moderation_result = 'review'
                 reasoning = f'Offensive content detected, needs review ({confidence_score:.2%} confidence)'
        
        else: # clean
             moderation_result = 'allowed'
             reasoning = f'Clean content ({confidence_score:.2%} confidence)'

        return {
            'sentiment': 'negative' if predicted_label in ['hate', 'offensive'] else 'neutral', # Legacy field support
            'model_label': predicted_label,
            'moderation_result': moderation_result,
            'confidence': round(confidence_score, 4),
            'reasoning': reasoning,
            'flagged_words': []
        }
    
    def _determine_moderation(self, sentiment: str, confidence: float) -> str:
        """
        Determine moderation action based on sentiment and confidence
        
        IMPORTANT: Negative sentiment ≠ Violation
        Only reject if contains actual toxic content (already filtered above)
        
        Args:
            sentiment: Sentiment label
            confidence: Confidence score
        
        Returns:
            Moderation label
        """
        # If we reach here, toxic words check already passed
        # So negative sentiment is just opinion/review, NOT a violation
        # Allow everything unless we're EXTREMELY uncertain (< 0.3)
        # This is a neutral platform - we should NOT censor opinions
        if confidence < 0.3:
            return 'review'
        else:
            return 'allowed'
    
    def _generate_reasoning(self, sentiment: str, moderation: str, confidence: float) -> str:
        """Generate reasoning explanation"""
        reasons = {
            ('positive', 'allowed'): 'Positive content, no violation',
            ('neutral', 'allowed'): 'Neutral content, no violation',
            ('negative', 'allowed'): 'Legitimate negative review, no violation',
            ('positive', 'review'): 'Content needs review (low confidence)',
            ('neutral', 'review'): 'Content needs review (low confidence)',
            ('negative', 'review'): 'Content needs review (low confidence)',
        }
        
        reason = reasons.get((sentiment, moderation), 'Unknown')
        
        if confidence < 0.3:
            reason += f" ({confidence:.2%} confidence)"
        
        return reason

    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Batch prediction (compatibility wrapper)
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
