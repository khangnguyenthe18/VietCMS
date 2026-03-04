"""
Multi-Task Inference Engine for Vietnamese Content Moderation

Supports:
- Multi-label classification (7 labels)
- Severity prediction (0-2)
- Span detection (optional)
- Rule-based fallback for edge cases
- Context-aware analysis (NEW)
- Semantic similarity checking (NEW)

Version: 2.0.0 - Enhanced Accuracy
Last Updated: 2025-12-19
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_phobert import MultiTaskPhoBERT
from nlp.preprocessing_advanced import preprocess_for_phobert, extract_pii, mask_pii
from nlp.taxonomy import (
    ModerationLabel, 
    SeverityLevel, 
    DEFAULT_LABELS, 
    combine_predictions,
    severity_to_action,
    LABEL_DESCRIPTIONS
)
from nlp.toxic_words import (
    get_critical_words,
    get_hate_speech_words,
    get_sexual_content_words,
    SEVERITY_SCORES,
    REJECT_THRESHOLD,
    REVIEW_THRESHOLD,
    AUTO_REJECT_CATEGORIES,
    ALLOWED_PHRASES,
)
from nlp.sentiment_words import HIGHLY_NEGATIVE, MODERATELY_NEGATIVE

# NEW: Import context analyzer for enhanced accuracy
try:
    from nlp.context_analyzer import (
        EnhancedModerationAnalyzer,
        ContextAnalyzer, 
        ContentIntent,
        get_enhanced_analyzer
    )
    HAS_CONTEXT_ANALYZER = True
except ImportError:
    HAS_CONTEXT_ANALYZER = False

# NEW: Import variant detector for obfuscation detection
try:
    from nlp.variant_detector import get_variant_detector, VariantDetector
    HAS_VARIANT_DETECTOR = True
except ImportError:
    HAS_VARIANT_DETECTOR = False

# NEW: Import 3-layer moderation pipeline (enhanced detection)
try:
    from nlp.text_normalizer import get_normalizer, VietnameseTextNormalizer
    from nlp.rule_checker import get_rule_checker, EnhancedRuleChecker
    HAS_THREE_LAYER_PIPELINE = True
except ImportError:
    HAS_THREE_LAYER_PIPELINE = False

logger = logging.getLogger(__name__)

if not HAS_CONTEXT_ANALYZER:
    logger.warning("Context analyzer not available, using basic mode")
if not HAS_VARIANT_DETECTOR:
    logger.warning("Variant detector not available, obfuscation detection disabled")
if HAS_THREE_LAYER_PIPELINE:
    logger.info("3-Layer Pipeline available: Enhanced normalization + rule-based detection enabled")
else:
    logger.warning("3-Layer Pipeline not available, using legacy detection")


class MultiTaskModerationInference:
    """
    Enhanced inference engine with multi-label and severity prediction
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        confidence_threshold: float = 0.5,
        use_rule_based_fallback: bool = True,
        use_context_analyzer: bool = True,  # NEW: Enable context-aware analysis
        use_variant_detector: bool = True   # NEW: Enable obfuscation detection
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_rule_based_fallback = use_rule_based_fallback
        self.use_context_analyzer = use_context_analyzer and HAS_CONTEXT_ANALYZER
        self.use_variant_detector = use_variant_detector and HAS_VARIANT_DETECTOR
        
        self.label_names = [label.value for label in DEFAULT_LABELS]
        self.num_labels = len(self.label_names)
        
        # Load model and tokenizer
        self.load_model()
        
        # NEW: Initialize context analyzer
        if self.use_context_analyzer:
            self.enhanced_analyzer = get_enhanced_analyzer()
            logger.info("Context analyzer enabled for enhanced accuracy")
        else:
            self.enhanced_analyzer = None
        
        # NEW: Initialize variant detector
        if self.use_variant_detector:
            self.variant_detector = get_variant_detector()
            logger.info("Variant detector enabled for obfuscation detection")
        else:
            self.variant_detector = None
        
        # NEW: Initialize 3-layer pipeline (preferred over legacy variant detector)
        self.use_three_layer_pipeline = HAS_THREE_LAYER_PIPELINE
        if self.use_three_layer_pipeline:
            self.text_normalizer = get_normalizer()
            self.enhanced_rule_checker = get_rule_checker()
            logger.info("3-Layer Pipeline initialized: text_normalizer + enhanced_rule_checker")
            
            # Initialize metrics counter
            self.metrics = {
                'total_processed': 0,
                'rule_based_catches': 0,
                'model_predictions': 0,
                'obfuscation_detected': 0,
                'hate_speech_detected': 0,
                'harassment_detected': 0,
                'rejected': 0,
                'reviewed': 0,
                'allowed': 0,
            }
        else:
            self.text_normalizer = None
            self.enhanced_rule_checker = None
            self.metrics = None
    
    def load_model(self):
        """Load trained multi-task model"""
        try:
            logger.info(f"Loading multi-task model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load task heads config
            task_heads_path = os.path.join(self.model_path, 'task_heads.pt')
            
            if os.path.exists(task_heads_path):
                checkpoint = torch.load(task_heads_path, map_location=self.device)
                config = checkpoint['config']
                
                # Initialize model
                self.model = MultiTaskPhoBERT(
                    model_name=self.model_path,
                    num_labels=config['num_labels'],
                    num_severity_levels=config['num_severity_levels'],
                    use_span_detection=config.get('use_span_detection', False)
                )
                
                # Load task head weights
                self.model.multi_label_classifier.load_state_dict(
                    checkpoint['multi_label_classifier']
                )
                self.model.severity_regressor.load_state_dict(
                    checkpoint['severity_regressor']
                )
                if config.get('use_span_detection') and checkpoint.get('span_detector'):
                    self.model.span_detector.load_state_dict(
                        checkpoint['span_detector']
                    )
                
                logger.info("Loaded trained multi-task model with task heads")
            else:
                # Fallback: load base model (untrained task heads)
                logger.warning("Task heads not found, loading base model")
                self.model = MultiTaskPhoBERT(
                    model_name=self.model_path,
                    num_labels=self.num_labels,
                    use_span_detection=False
                )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _is_in_allowed_phrase(self, text_lower: str, word: str) -> bool:
        """
        NEW: Check if a word appears in an allowed phrase context
        """
        for phrase in ALLOWED_PHRASES:
            if phrase in text_lower:
                # Check if the word is part of this allowed phrase
                if word in phrase.lower():
                    return True
        return False
    
    def _detect_variants(self, text: str) -> Tuple[List[str], bool, str]:
        """
        NEW: Detect obfuscated variants of toxic words
        
        Returns:
            (detected_words, has_obfuscation, normalized_text)
        """
        if not self.variant_detector:
            return [], False, text.lower()
        
        try:
            result = self.variant_detector.analyze(text)
            
            detected = []
            for variant in result.get('detected_variants', []):
                detected.append(variant.get('normalized', ''))
            
            has_obfuscation = result.get('has_obfuscation', False)
            normalized = result.get('normalized_text', text.lower())
            
            if detected:
                logger.info(f"Variant detection found: {detected} (obfuscation: {has_obfuscation})")
            
            return detected, has_obfuscation, normalized
            
        except Exception as e:
            logger.error(f"Variant detection error: {e}")
            return [], False, text.lower()

    def _filter_safe_context_words(self, text: str, detected_words: List[str]) -> List[str]:
        """
        NEW: Filter out words that appear in safe/allowed contexts
        """
        text_lower = text.lower()
        filtered = []
        
        for word in detected_words:
            # Check allowed phrases
            if self._is_in_allowed_phrase(text_lower, word):
                logger.debug(f"Word '{word}' in allowed phrase context, skipping")
                continue
            
            # Check context analyzer if available
            if self.enhanced_analyzer:
                if self.enhanced_analyzer.context_analyzer.is_safe_context(text, word):
                    logger.debug(f"Word '{word}' in safe context, skipping")
                    continue
            
            filtered.append(word)
        
        return filtered

    def _enhanced_rule_check(self, text: str) -> Optional[Dict[str, Any]]:
        """
        NEW: 3-Layer Pipeline check using text_normalizer + enhanced_rule_checker
        
        This method:
        1. Normalizes text (Layer A)
        2. Runs enhanced rule check (Layer B)
        3. Returns result in standard format
        """
        if not self.text_normalizer or not self.enhanced_rule_checker:
            return None
        
        try:
            # Layer A: Normalize text
            versions = self.text_normalizer.create_all_versions(text)
            original = versions['original']
            normalized = versions['fully_normalized']
            no_diacritics = versions['no_diacritics']
            metadata = versions['metadata']
            
            # Log obfuscation if detected
            if metadata.get('has_obfuscation'):
                obf_types = metadata.get('obfuscation_types', [])
                logger.info(f"3-Layer: Obfuscation detected: {obf_types}")
            
            # Layer B: Enhanced rule check
            result = self.enhanced_rule_checker.check(
                text=original,
                normalized_text=normalized,
                no_diacritics_text=no_diacritics,
                metadata=metadata
            )
            
            if not result:
                return None
            
            # Map to standard format with SeverityLevel
            action = result.get('action', 'allowed')
            labels = result.get('labels', [])
            findings = result.get('findings', [])
            
            # Determine severity level
            if action == 'reject':
                severity = SeverityLevel.SEVERE
            elif action == 'review':
                severity = SeverityLevel.MODERATE
            else:
                severity = SeverityLevel.SAFE
            
            # Build standard return format
            return {
                'labels': labels,
                'severities': [severity],
                'action': action,
                'confidence': result.get('confidence', 0.9),
                'reasoning': result.get('reasoning', ''),
                'flagged_words': [f.get('matched', '') for f in findings[:5]],
                'method': 'three_layer_pipeline',
                'has_obfuscation': metadata.get('has_obfuscation', False),
                'obfuscation_types': metadata.get('obfuscation_types', []),
                'normalized_text': normalized,
                'findings': findings,
            }
            
        except Exception as e:
            logger.error(f"3-Layer Pipeline check error: {e}")
            return None
    
    def rule_based_check(self, text: str) -> Optional[Dict[str, Any]]:
        """
        ENHANCED Rule-based pre-check with multi-category detection
        NOW with 3-layer pipeline integration (preferred) + legacy fallback
        """
        # Update metrics
        if self.metrics:
            self.metrics['total_processed'] += 1
        
        # ===== NEW: 3-LAYER PIPELINE CHECK (PREFERRED) =====
        if self.use_three_layer_pipeline:
            result = self._enhanced_rule_check(text)
            if result:
                # Update metrics based on result
                if self.metrics:
                    self.metrics['rule_based_catches'] += 1
                    if result.get('has_obfuscation'):
                        self.metrics['obfuscation_detected'] += 1
                    if 'hate' in result.get('labels', []) or 'racism' in result.get('labels', []):
                        self.metrics['hate_speech_detected'] += 1
                    if 'harassment' in result.get('labels', []):
                        self.metrics['harassment_detected'] += 1
                    if result.get('action') == 'reject':
                        self.metrics['rejected'] += 1
                    elif result.get('action') == 'review':
                        self.metrics['reviewed'] += 1
                    else:
                        self.metrics['allowed'] += 1
                return result
        
        # ===== LEGACY: VARIANT DETECTION FALLBACK =====
        text_lower = text.lower()
        
        # Detect obfuscated variants first
        variant_words, has_obfuscation, normalized_text = self._detect_variants(text)
        
        if variant_words and has_obfuscation:
            # If obfuscation detected with toxic content, higher severity
            logger.info(f"Obfuscation detected with toxic content: {variant_words}")
            return {
                'labels': ['toxicity', 'obfuscation'],
                'severities': [SeverityLevel.SEVERE],
                'action': 'reject',
                'confidence': 0.95,
                'reasoning': f' Bypass filter detected: {", ".join(variant_words[:3])} (obfuscation)',
                'flagged_words': variant_words,
                'method': 'rule_based_variant_detection',
                'has_obfuscation': True
            }
        elif variant_words:
            # Add to text_lower for further checking
            text_lower = normalized_text
        
        # ===== 1. CONTEXT PRE-ANALYSIS =====
        context_result = None
        if self.enhanced_analyzer:
            context_result = self.enhanced_analyzer.context_analyzer.analyze(text)
            
            # If it's legitimate criticism targeting product, be more lenient
            if context_result.is_legitimate_criticism:
                logger.info(f"Legitimate criticism detected, applying lenient rules")
        
        # ===== 1. CHECK PII FIRST =====
        pii = extract_pii(text)
        has_pii = any(len(v) > 0 for v in pii.values())
        
        if has_pii:
            return {
                'labels': ['pii'],
                'severities': [SeverityLevel.MODERATE],
                'action': 'review',
                'confidence': 0.95,
                'reasoning': f'PII detected: {", ".join([k for k, v in pii.items() if v])}',
                'pii_detected': pii,
                'method': 'rule_based_pii'
            }
        
        # ===== 2. CHECK HATE SPEECH (HIGHEST PRIORITY) =====
        hate_speech_words = get_hate_speech_words()
        detected_hate = []
        
        for word in hate_speech_words:
            if word in text_lower:
                detected_hate.append(word)
        
        # NEW: Filter out safe context words
        detected_hate = self._filter_safe_context_words(text, detected_hate)
        
        if detected_hate:
            # Double check with context analyzer for hate speech
            if context_result and context_result.intent.value == 'hate_speech':
                return {
                    'labels': ['hate'],
                    'severities': [SeverityLevel.SEVERE],
                    'action': 'reject',
                    'confidence': 0.98,
                    'reasoning': f' HATE SPEECH confirmed: {", ".join(detected_hate[:3])}',
                    'flagged_words': detected_hate,
                    'method': 'rule_based_hate_speech_confirmed'
                }
            elif len(detected_hate) >= 2 or any(len(w) > 5 for w in detected_hate):
                # Multiple hate words or long hate phrases
                return {
                    'labels': ['hate'],
                    'severities': [SeverityLevel.SEVERE],
                    'action': 'reject',
                    'confidence': 0.95,
                    'reasoning': f' HATE SPEECH: {", ".join(detected_hate[:3])}',
                    'flagged_words': detected_hate,
                    'method': 'rule_based_hate_speech'
                }
            else:
                # Single short hate word - need review
                return {
                    'labels': ['hate'],
                    'severities': [SeverityLevel.MODERATE],
                    'action': 'review',
                    'confidence': 0.75,
                    'reasoning': f' Potential hate speech: {", ".join(detected_hate[:3])}',
                    'flagged_words': detected_hate,
                    'method': 'rule_based_hate_speech_review'
                }
        
        # ===== 3. CHECK SEXUAL CONTENT =====
        sexual_words = get_sexual_content_words()
        detected_sexual = []
        
        for word in sexual_words:
            if word in text_lower:
                detected_sexual.append(word)
        
        # NEW: Filter out safe context words
        detected_sexual = self._filter_safe_context_words(text, detected_sexual)
        
        if detected_sexual:
            # Auto-reject explicit sexual content
            return {
                'labels': ['sexual'],
                'severities': [SeverityLevel.SEVERE],
                'action': 'reject',
                'confidence': 0.9,
                'reasoning': f' SEXUAL CONTENT: {", ".join(detected_sexual[:3])}',
                'flagged_words': detected_sexual,
                'method': 'rule_based_sexual'
            }
        
        # ===== 4. CHECK CRITICAL TOXIC WORDS =====
        critical_words = get_critical_words()
        detected_critical = []
        
        for word in critical_words:
            if word in text_lower:
                detected_critical.append(word)
        
        # NEW: Filter out safe context words
        detected_critical = self._filter_safe_context_words(text, detected_critical)
        
        if detected_critical:
            # NEW: Apply context-aware decision
            if context_result:
                severity_modifier = context_result.severity_modifier
                
                # If it's product feedback with toxic words, reduce severity
                if context_result.is_legitimate_criticism and severity_modifier < 0.7:
                    return {
                        'labels': ['profanity'],
                        'severities': [SeverityLevel.MILD],
                        'action': 'review',  # Review instead of reject
                        'confidence': 0.7,
                        'reasoning': f' Strong language in feedback: {", ".join(detected_critical[:3])}',
                        'flagged_words': detected_critical,
                        'method': 'rule_based_feedback_profanity',
                        'context_modifier': severity_modifier
                    }
                elif context_result.targets_person:
                    # Targets person = more severe
                    return {
                        'labels': ['toxicity', 'harassment'],
                        'severities': [SeverityLevel.SEVERE],
                        'action': 'reject',
                        'confidence': 0.95,
                        'reasoning': f' Personal attack: {", ".join(detected_critical[:3])}',
                        'flagged_words': detected_critical,
                        'method': 'rule_based_personal_attack'
                    }
            
            # Default: Very severe violation
            return {
                'labels': ['toxicity'],
                'severities': [SeverityLevel.SEVERE],
                'action': 'reject',
                'confidence': 0.9,
                'reasoning': f'Severe violation: {", ".join(detected_critical[:3])}',
                'flagged_words': detected_critical,
                'method': 'rule_based_toxicity'
            }
        
        # No clear rule-based violation
        return None
    
    def predict(self, text: str, return_spans: bool = False) -> Dict[str, Any]:
        """
        Predict moderation labels and severity
        
        Args:
            text: Input Vietnamese text
            return_spans: Whether to return span predictions
            
        Returns:
            Dict with labels, severities, action, confidence, reasoning
        """
        # Step 1: Rule-based pre-check
        if self.use_rule_based_fallback:
            rule_result = self.rule_based_check(text)
            if rule_result is not None:
                return rule_result
        
        # Step 2: Preprocess
        processed_text, metadata = preprocess_for_phobert(text)
        
        # Check if preprocessing detected obfuscations
        if metadata.get('obfuscations'):
            logger.info(f"Detected obfuscations: {metadata['obfuscations']}")
        
        # Step 3: Tokenize
        inputs = self.tokenizer(
            processed_text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Step 4: Model inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                threshold=self.confidence_threshold
            )
        
        # Step 5: Parse predictions
        multi_label_preds = predictions['multi_label_preds'][0].cpu().numpy()  # [num_labels]
        multi_label_probs = predictions['multi_label_probs'][0].cpu().numpy()  # [num_labels]
        severity_pred = predictions['severity_preds'][0].item()
        severity_score = predictions['severity_scores'][0].item()
        
        # Get triggered labels
        triggered_indices = np.where(multi_label_preds == 1)[0]
        triggered_labels = [self.label_names[i] for i in triggered_indices]
        triggered_probs = [float(multi_label_probs[i]) for i in triggered_indices]

        # If no labels triggered, it's clean
        if not triggered_labels:
            return {
                'labels': [],
                'severities': [],
                'action': 'allowed',
                'confidence': float(1 - multi_label_probs.max()),
                'reasoning': 'Clean content, no violation',
                'all_probabilities': {
                    label: float(prob)
                    for label, prob in zip(self.label_names, multi_label_probs)
                },
                'method': 'ml_model'
            }
        
        # Filter: Only block truly harmful content (toxic, hate, harassment)
        # Allow negative feedback/complaints - those are valid customer opinions
        harmful_labels = {'toxicity', 'hate', 'harassment', 'threat', 'pii', 'sexual'}
        triggered_harmful = [l for l in triggered_labels if l in harmful_labels]
        
        # If only mild labels (like 'spam' or 'profanity' with low confidence), allow
        if not triggered_harmful:
            # Check if any profanity with high confidence
            profanity_idx = self.label_names.index('profanity') if 'profanity' in self.label_names else -1
            if profanity_idx >= 0 and multi_label_preds[profanity_idx] == 1:
                if multi_label_probs[profanity_idx] < 0.8:  # Not very confident
                    return {
                        'labels': [],
                        'severities': [],
                        'action': 'allowed',
                        'confidence': 0.6,
                        'reasoning': 'Strong language but no severe violation',
                        'method': 'ml_model_filtered'
                    }
            else:
                return {
                    'labels': [],
                    'severities': [],
                    'action': 'allowed',
                    'confidence': 0.7,
                    'reasoning': 'Negative but valid feedback (customer opinion)',
                    'method': 'ml_model_filtered'
                }
        
        # Map to action based on severity
        action = severity_to_action(severity_pred)
        
        # Generate reasoning
        reasoning_parts = []
        for label, prob in zip(triggered_labels, triggered_probs):
            label_desc = LABEL_DESCRIPTIONS.get(
                ModerationLabel(label), {}
            ).get('en', label)
            reasoning_parts.append(f"{label_desc} ({prob:.2%})")
        
        reasoning = f"Violation detected: {', '.join(reasoning_parts)} | Severity: {severity_pred}"
        
        # Build result
        result = {
            'labels': triggered_labels,
            'severities': [severity_pred] * len(triggered_labels),
            'probabilities': {label: prob for label, prob in zip(triggered_labels, triggered_probs)},
            'action': action,
            'confidence': float(np.mean(triggered_probs)) if triggered_probs else 0.5,
            'reasoning': reasoning,
            'severity_score': float(severity_score),
            'all_probabilities': {
                label: float(prob) 
                for label, prob in zip(self.label_names, multi_label_probs)
            },
            'method': 'ml_model'
        }
        
        # Add span predictions if requested
        if return_spans and predictions['span_preds'] is not None:
            span_preds = predictions['span_preds'][0].cpu().numpy()
            result['span_predictions'] = self._extract_spans(
                processed_text, span_preds, inputs['attention_mask'][0]
            )
        
        return result
    
    def _extract_spans(
        self, 
        text: str, 
        span_preds: np.ndarray, 
        attention_mask: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Extract violation spans from token-level predictions
        
        Args:
            text: Processed text
            span_preds: [seq_len] token predictions (0 or 1)
            attention_mask: [seq_len] attention mask
            
        Returns:
            List of span dicts with start, end, text
        """
        # Get valid tokens (not padding)
        valid_length = attention_mask.sum().item()
        valid_preds = span_preds[:valid_length]
        
        # Tokenize to get token-to-char mapping
        tokens = self.tokenizer.tokenize(text)
        
        spans = []
        current_span = None
        
        for i, (token, pred) in enumerate(zip(tokens, valid_preds)):
            if pred == 1:  # Violation token
                if current_span is None:
                    current_span = {
                        'start': i,
                        'tokens': [token]
                    }
                else:
                    current_span['tokens'].append(token)
            else:
                if current_span is not None:
                    # End current span
                    current_span['end'] = i
                    current_span['text'] = ' '.join(current_span['tokens'])
                    spans.append(current_span)
                    current_span = None
        
        # Close last span if exists
        if current_span is not None:
            current_span['end'] = len(tokens)
            current_span['text'] = ' '.join(current_span['tokens'])
            spans.append(current_span)
        
        return spans
    
    def batch_predict(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        True Batch prediction for multiple texts
        Significantly faster than looping predict()
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing (internal chunking)
            
        Returns:
            List of prediction dicts
        """
        results = []
        
        # Process in chunks to avoid OOM on very large lists
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 1. Rule-based checks (must be done individually)
            batch_results = [None] * len(batch_texts)
            indices_to_predict = []
            texts_to_predict = []
            
            if self.use_rule_based_fallback:
                for idx, text in enumerate(batch_texts):
                    rule_result = self.rule_based_check(text)
                    if rule_result:
                        batch_results[idx] = rule_result
                    else:
                        indices_to_predict.append(idx)
                        texts_to_predict.append(text)
            else:
                indices_to_predict = list(range(len(batch_texts)))
                texts_to_predict = batch_texts
            
            # If everything was handled by rules, continue
            if not texts_to_predict:
                results.extend(batch_results)
                continue
                
            # 2. Preprocess
            processed_texts = [preprocess_for_phobert(t)[0] for t in texts_to_predict]
            
            # 3. Tokenize batch
            inputs = self.tokenizer(
                processed_texts,
                max_length=256,
                padding=True,  # Pad to longest in batch
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 4. Model inference (Single call!)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model.predict(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    threshold=self.confidence_threshold
                )
            
            # 5. Parse predictions
            # Move all to CPU/numpy at once
            multi_label_preds = predictions['multi_label_preds'].cpu().numpy()
            multi_label_probs = predictions['multi_label_probs'].cpu().numpy()
            severity_preds = predictions['severity_preds'].cpu().numpy()
            severity_scores = predictions['severity_scores'].cpu().numpy()
            
            # Map back to results
            for j, original_idx in enumerate(indices_to_predict):
                # Extract single item results
                item_preds = multi_label_preds[j]
                item_probs = multi_label_probs[j]
                item_severity = severity_preds[j]
                item_severity_score = severity_scores[j]
                
                # Logic from predict()
                triggered_indices = np.where(item_preds == 1)[0]
                triggered_labels = [self.label_names[k] for k in triggered_indices]
                triggered_probs = [float(item_probs[k]) for k in triggered_indices]
                
                # Build result dict
                result = {}
                
                if not triggered_labels:
                    result = {
                        'labels': [],
                        'severities': [],
                        'action': 'allowed',
                        'confidence': float(1 - item_probs.max()),
                        'reasoning': 'Clean content, no violation',
                        'method': 'ml_model_batch'
                    }
                else:
                    # Filter logic
                    harmful_labels = {'toxicity', 'hate', 'harassment', 'threat', 'pii', 'sexual'}
                    triggered_harmful = [l for l in triggered_labels if l in harmful_labels]
                    
                    if not triggered_harmful:
                        # Check profanity confidence
                        profanity_idx = self.label_names.index('profanity') if 'profanity' in self.label_names else -1
                        if profanity_idx >= 0 and item_preds[profanity_idx] == 1 and item_probs[profanity_idx] < 0.8:
                            result = {
                                'labels': [],
                                'severities': [],
                                'action': 'allowed',
                                'confidence': 0.6,
                                'reasoning': 'Strong language but no severe violation',
                                'method': 'ml_model_filtered'
                            }
                        else:
                            # Check if just negative sentiment/spam
                            result = {
                                'labels': [],
                                'severities': [],
                                'action': 'allowed',
                                'confidence': 0.7,
                                'reasoning': 'Negative but valid feedback',
                                'method': 'ml_model_filtered'
                            }
                    
                    # If still empty (filtered out), set it
                    if not result:
                        action = severity_to_action(item_severity)
                        reasoning_parts = []
                        for label, prob in zip(triggered_labels, triggered_probs):
                            label_desc = LABEL_DESCRIPTIONS.get(ModerationLabel(label), {}).get('en', label)
                            reasoning_parts.append(f"{label_desc} ({prob:.2%})")
                        
                        reasoning = f"Violation detected: {', '.join(reasoning_parts)} | Severity: {item_severity}"
                        
                        result = {
                            'labels': triggered_labels,
                            'severities': [int(item_severity)] * len(triggered_labels),
                            'probabilities': {l: p for l, p in zip(triggered_labels, triggered_probs)},
                            'action': action,
                            'confidence': float(np.mean(triggered_probs)),
                            'reasoning': reasoning,
                            'severity_score': float(item_severity_score),
                            'method': 'ml_model_batch'
                        }
                
                # Add all probs
                result['all_probabilities'] = {
                    label: float(prob) 
                    for label, prob in zip(self.label_names, item_probs)
                }
                
                batch_results[original_idx] = result
            
            results.extend(batch_results)
        
        return results


# Backward compatibility wrapper
class ModerationInference:
    """
    Wrapper for backward compatibility with existing code
    Maps multi-label output to single sentiment/moderation result
    """
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model_path = model_path or 'vinai/phobert-base-v2'
        self.device = device
        
        # Try to load multi-task model, fallback to rule-based
        try:
            self.engine = MultiTaskModerationInference(
                model_path=self.model_path,
                device=device
            )
            self.use_ml = True
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}, using rule-based only")
            self.use_ml = False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict with backward compatible output format
        
        Returns:
            Dict with sentiment, moderation_result, confidence, reasoning
        """
        if self.use_ml:
            result = self.engine.predict(text)
            
            # Map to old format
            action_map = {
                'allowed': 'allowed',
                'review': 'review',
                'reject': 'reject'
            }
            
            # Determine sentiment from labels
            if 'toxicity' in result['labels'] or 'hate' in result['labels']:
                sentiment = 'negative'
            elif not result['labels']:
                sentiment = 'positive'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'moderation_result': action_map.get(result['action'], 'review'),
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'labels': result['labels'],  # Extra info
                'flagged_words': []
            }
        else:
            # Fallback to simple rule-based
            from nlp.inference import ModerationInference as OldInference
            old_engine = OldInference(model_path=self.model_path, device=self.device)
            return old_engine.predict(text)
    
    # ==================== METRICS METHODS ====================
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for monitoring.
        
        Returns:
            Dictionary with all metric counters and calculated rates
        """
        if not self.metrics:
            return {'error': 'Metrics not available (3-layer pipeline not initialized)'}
        
        total = self.metrics['total_processed']
        if total == 0:
            return {**self.metrics, 'rates': {}}
        
        rates = {
            'rule_based_catch_rate': self.metrics['rule_based_catches'] / total * 100,
            'obfuscation_rate': self.metrics['obfuscation_detected'] / total * 100,
            'hate_speech_rate': self.metrics['hate_speech_detected'] / total * 100,
            'harassment_rate': self.metrics['harassment_detected'] / total * 100,
            'rejection_rate': self.metrics['rejected'] / total * 100,
            'review_rate': self.metrics['reviewed'] / total * 100,
            'approval_rate': self.metrics['allowed'] / total * 100,
        }
        
        return {**self.metrics, 'rates': rates}
    
    def reset_metrics(self):
        """Reset all metrics counters to zero"""
        if self.metrics:
            for key in self.metrics:
                self.metrics[key] = 0
            logger.info("Metrics reset to zero")
    
    def log_metrics(self):
        """Log current metrics summary"""
        metrics = self.get_metrics()
        
        if 'error' in metrics:
            logger.warning(metrics['error'])
            return
        
        logger.info("=" * 60)
        logger.info("MODERATION METRICS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total processed: {metrics['total_processed']}")
        logger.info(f"Rule-based catches: {metrics['rule_based_catches']} ({metrics['rates'].get('rule_based_catch_rate', 0):.1f}%)")
        logger.info(f"Obfuscation detected: {metrics['obfuscation_detected']} ({metrics['rates'].get('obfuscation_rate', 0):.1f}%)")
        logger.info(f"Hate speech: {metrics['hate_speech_detected']} ({metrics['rates'].get('hate_speech_rate', 0):.1f}%)")
        logger.info(f"Harassment: {metrics['harassment_detected']} ({metrics['rates'].get('harassment_rate', 0):.1f}%)")
        logger.info("-" * 40)
        logger.info(f"Actions: Rejected={metrics['rejected']} | Review={metrics['reviewed']} | Allowed={metrics['allowed']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test inference
    test_texts = [
        "Sản phẩm rất tốt, tôi rất hài lòng!",
        "Đồ rác vãi lồn, không mua nữa",
        "Shop lừa đảo, inbox mua hàng 0123456789",
        "Bọn khỉ đen này ngu vãi",
        "Sản phẩm bình thường, giá hơi cao",
    ]
    
    print("="*80)
    print("MULTI-TASK INFERENCE TEST")
    print("="*80)
    
    # Test with base model (not trained)
    engine = MultiTaskModerationInference(
        model_path='vinai/phobert-base-v2',
        device='cpu'
    )
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = engine.predict(text)
        print(f"Labels: {result['labels']}")
        print(f"Action: {result['action']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Reasoning: {result['reasoning']}")
        if 'all_probabilities' in result:
            print("All probabilities:")
            for label, prob in result['all_probabilities'].items():
                if prob > 0.3:
                    print(f"  {label}: {prob:.2%}")

