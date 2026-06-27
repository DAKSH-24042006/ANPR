"""Post-processing Module for ANPR Inference Pipeline.

Handles alphanumeric text cleaning, OCR confidence threshold checks,
Indian registration format validations, and state-specific character substitutions.
"""

from src import config
from src.utils import clean_ocr_text, is_valid_indian_plate, DIGIT_TO_ALPHA, ALPHA_TO_DIGIT

def correct_characters_by_position(clean_text: str) -> str:
    """Applies position-based corrections based on standard Indian license plate structures.

    Indian plates typically follow: State (2 letters) + District (1 or 2 digits)
    + Series (1 to 3 letters) + Unique number (4 digits).
    We attempt corrections assuming both 1-digit and 2-digit district codes,
    returning the one that successfully validates against the regex.

    Args:
        clean_text: Alphanumeric uppercase string.

    Returns:
        Corrected registration string.
    """
    if len(clean_text) < 7:
        return clean_text

    def apply_substitution(text_str: str, district_len: int) -> str:
        # 1. State code (First 2 characters: ALWAYS letters)
        state = list(text_str[:2])
        for i in range(2):
            if state[i] in DIGIT_TO_ALPHA:
                state[i] = DIGIT_TO_ALPHA[state[i]]

        # 2. District code (Next 1 or 2 characters: ALWAYS digits)
        district = list(text_str[2:2 + district_len])
        for i in range(len(district)):
            if district[i] in ALPHA_TO_DIGIT:
                district[i] = ALPHA_TO_DIGIT[district[i]]

        # 3. Unique number (Last 4 characters: ALWAYS digits)
        end_digits = list(text_str[-4:])
        for i in range(4):
            if end_digits[i] in ALPHA_TO_DIGIT:
                end_digits[i] = ALPHA_TO_DIGIT[end_digits[i]]

        # 4. Series letters (Remaining middle characters: ALWAYS letters)
        series_start = 2 + district_len
        series_end = len(text_str) - 4
        
        # If the structure is invalid (e.g. overlaps), return raw text
        if series_start >= series_end:
            return text_str
            
        series = list(text_str[series_start:series_end])
        for i in range(len(series)):
            if series[i] in DIGIT_TO_ALPHA:
                series[i] = DIGIT_TO_ALPHA[series[i]]

        return "".join(state + district + series + end_digits)

    # Attempt 2-digit district code format correction (e.g. MH12PQ1234)
    candidate_2 = apply_substitution(clean_text, district_len=2)
    if is_valid_indian_plate(candidate_2):
        return candidate_2

    # Attempt 1-digit district code format correction (e.g. DL3CAY5678)
    candidate_1 = apply_substitution(clean_text, district_len=1)
    if is_valid_indian_plate(candidate_1):
        return candidate_1

    # Fallback to the 2-digit candidate (most common) if none match the validation regex
    return candidate_2


def post_process_ocr(raw_text: str, confidence: float) -> tuple:
    """Cleans, validates, and corrects the raw license plate text.

    Uses a sliding window to search for standard Indian registration plate
    substrings within the OCR output to isolate numbers from brand or border text.

    Args:
        raw_text: Raw string output from PaddleOCR.
        confidence: Combined mean OCR confidence score.

    Returns:
        A tuple of:
            - is_valid (bool): True if a valid registration number was isolated and corrected.
            - clean_text (str): Cleaned and character-corrected registration number.
    """
    # 1. Reject text below config threshold
    if confidence < config.OCR_CONF_THRESHOLD:
        return False, ""

    # 2. Clean text (remove spaces, symbols and uppercase)
    cleaned = clean_ocr_text(raw_text)
    if not cleaned:
        return False, ""

    # 3. Sliding window substring scan to extract standard Indian plates (lengths 7 to 10)
    n = len(cleaned)
    for length in range(min(10, n), 6, -1):
        for start in range(n - length + 1):
            sub = cleaned[start:start+length]
            corrected_sub = correct_characters_by_position(sub)
            if is_valid_indian_plate(corrected_sub):
                return True, corrected_sub

    # 4. Fallback if no clean substring matches the regex
    fallback_corrected = correct_characters_by_position(cleaned)
    is_valid = is_valid_indian_plate(fallback_corrected)
    
    return is_valid, fallback_corrected
