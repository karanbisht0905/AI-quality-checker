"""
AI Response Quality Checker
Rule-based Text Analysis Tool
Author: Karan Bisht
"""

import re
import json
import math
from collections import Counter


# ──────────────────────────────────────────────
#  RULE SETS
# ──────────────────────────────────────────────

FILLER_WORDS = [
    "basically", "literally", "actually", "just", "very", "really",
    "quite", "rather", "somewhat", "kind of", "sort of", "you know",
    "i mean", "like", "obviously", "clearly", "simply", "totally",
    "absolutely", "definitely", "certainly", "honestly", "frankly"
]

VAGUE_WORDS = [
    "thing", "stuff", "things", "etc", "and so on", "and so forth",
    "various", "several", "many", "some", "a lot", "lots",
    "something", "somehow", "somewhere", "sometime"
]

UNCERTAINTY_PHRASES = [
    "i think", "i believe", "i'm not sure", "i'm not certain",
    "maybe", "perhaps", "possibly", "probably", "might be",
    "could be", "i guess", "i suppose", "not sure if"
]

OVERCONFIDENCE_PHRASES = [
    "always", "never", "every", "all", "none", "without exception",
    "100%", "guaranteed", "definitely will", "certainly will",
    "impossible", "must be", "have to be"
]

PASSIVE_VOICE_INDICATORS = [
    r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b', r'\bis\s+\w+ed\b',
    r'\bare\s+\w+ed\b', r'\bbeen\s+\w+ed\b', r'\bbeing\s+\w+ed\b',
    r'\bwas\s+\w+en\b', r'\bwere\s+\w+en\b', r'\bis\s+\w+en\b'
]

TRANSITION_WORDS = [
    "however", "therefore", "furthermore", "moreover", "additionally",
    "consequently", "nevertheless", "nonetheless", "meanwhile",
    "subsequently", "in contrast", "on the other hand", "as a result",
    "in conclusion", "in summary", "first", "second", "finally",
    "for example", "for instance", "such as", "in addition"
]

POSITIVE_SENTIMENT = [
    "great", "excellent", "good", "helpful", "useful", "effective",
    "clear", "accurate", "correct", "right", "perfect", "well",
    "best", "better", "improved", "success"
]

NEGATIVE_SENTIMENT = [
    "bad", "wrong", "incorrect", "poor", "terrible", "awful",
    "useless", "unhelpful", "confusing", "unclear", "misleading",
    "inaccurate", "false", "error", "mistake", "fail"
]

TECHNICAL_JARGON = [
    "algorithm", "neural", "model", "inference", "parameter", "token",
    "embedding", "vector", "matrix", "gradient", "optimization",
    "architecture", "transformer", "attention", "dataset", "training"
]


# ──────────────────────────────────────────────
#  ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def get_words(text):
    return re.findall(r'\b[a-zA-Z]+\b', text)


# 1. Grammar & Punctuation
def check_grammar(text):
    issues = []
    score = 100

    # Double spaces
    if '  ' in text:
        issues.append("Double spaces detected")
        score -= 5

    # Sentence starts with lowercase (after period)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for i, s in enumerate(sentences[1:], 1):
        if s and s[0].islower() and s[0].isalpha():
            issues.append(f"Sentence #{i+1} starts with lowercase")
            score -= 8

    # Missing space after punctuation
    if re.search(r'[,\.!?][A-Za-z]', text):
        issues.append("Missing space after punctuation")
        score -= 6

    # Repeated punctuation (not ellipsis)
    if re.search(r'[!?]{2,}|,{2,}', text):
        issues.append("Repeated punctuation marks")
        score -= 5

    # Very long sentence (>60 words)
    for i, sent in enumerate(get_sentences(text)):
        if len(sent.split()) > 60:
            issues.append(f"Sentence #{i+1} is too long ({len(sent.split())} words)")
            score -= 7

    # Unmatched parentheses/brackets
    if text.count('(') != text.count(')'):
        issues.append("Unmatched parentheses")
        score -= 5

    return max(0, score), issues


# 2. Clarity Score
def check_clarity(text):
    issues = []
    score = 100
    words = get_words(text)
    lower = text.lower()

    # Filler words
    found_fillers = [fw for fw in FILLER_WORDS if fw in lower]
    if found_fillers:
        penalty = min(30, len(found_fillers) * 5)
        score -= penalty
        issues.append(f"Filler words found: {', '.join(found_fillers[:5])}")

    # Vague words
    found_vague = [vw for vw in VAGUE_WORDS if re.search(r'\b' + vw + r'\b', lower)]
    if found_vague:
        penalty = min(20, len(found_vague) * 4)
        score -= penalty
        issues.append(f"Vague language: {', '.join(found_vague[:5])}")

    # Passive voice
    passive_count = sum(1 for pat in PASSIVE_VOICE_INDICATORS if re.search(pat, lower))
    if passive_count > 2:
        score -= min(15, passive_count * 3)
        issues.append(f"High passive voice usage ({passive_count} instances)")

    # Avg word length (complex words = less clear)
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len > 7:
            score -= 10
            issues.append(f"Complex vocabulary (avg word length: {avg_len:.1f} chars)")

    # Transition words (good for clarity)
    found_transitions = [t for t in TRANSITION_WORDS if t in lower]
    if len(found_transitions) >= 2:
        score += min(10, len(found_transitions) * 2)

    return max(0, min(100, score)), issues


# 3. Tone Analysis
def check_tone(text):
    lower = text.lower()
    words = tokenize(text)

    pos_count = sum(1 for w in POSITIVE_SENTIMENT if w in lower)
    neg_count = sum(1 for w in NEGATIVE_SENTIMENT if w in lower)
    uncertain_count = sum(1 for p in UNCERTAINTY_PHRASES if p in lower)
    overconf_count = sum(1 for p in OVERCONFIDENCE_PHRASES if p in lower)

    total = pos_count + neg_count + 1
    sentiment_ratio = pos_count / total

    if sentiment_ratio > 0.7:
        tone = "Positive"
    elif sentiment_ratio < 0.3:
        tone = "Negative"
    else:
        tone = "Neutral"

    if uncertain_count > 2:
        tone += " / Uncertain"
    if overconf_count > 2:
        tone += " / Overconfident"

    # Score: neutral/balanced is best for AI responses
    score = 70
    if tone == "Neutral":
        score = 95
    elif "Positive" in tone and "Overconfident" not in tone:
        score = 80
    elif "Negative" in tone:
        score = 50
    if "Uncertain" in tone:
        score -= 15
    if "Overconfident" in tone:
        score -= 20

    details = {
        "tone": tone,
        "positive_signals": pos_count,
        "negative_signals": neg_count,
        "uncertainty_phrases": uncertain_count,
        "overconfidence_phrases": overconf_count
    }
    return max(0, score), details


# 4. Contextual Relevance (vs expected/prompt)
def check_relevance(response, expected_keywords):
    if not expected_keywords:
        return 100, {"note": "No expected keywords provided"}

    response_lower = response.lower()
    keywords = [kw.strip().lower() for kw in expected_keywords.split(',') if kw.strip()]

    if not keywords:
        return 100, {"note": "No keywords to check"}

    matched = [kw for kw in keywords if kw in response_lower]
    missing = [kw for kw in keywords if kw not in response_lower]

    score = int((len(matched) / len(keywords)) * 100)

    return score, {
        "total_keywords": len(keywords),
        "matched": matched,
        "missing": missing,
        "coverage": f"{score}%"
    }


# 5. Readability (Flesch-Kincaid style)
def check_readability(text):
    sentences = get_sentences(text)
    words = get_words(text)

    if not sentences or not words:
        return 0, {"error": "Insufficient text"}

    num_sentences = len(sentences)
    num_words = len(words)

    # Count syllables (approximation)
    def count_syllables(word):
        word = word.lower()
        count = len(re.findall(r'[aeiou]+', word))
        if word.endswith('e') and len(word) > 2:
            count = max(1, count - 1)
        return max(1, count)

    total_syllables = sum(count_syllables(w) for w in words)

    avg_sent_len = num_words / num_sentences
    avg_syllables = total_syllables / num_words

    # Flesch Reading Ease
    fre = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syllables)
    fre = max(0, min(100, fre))

    if fre >= 70:
        level = "Easy (Good)"
    elif fre >= 50:
        level = "Moderate"
    elif fre >= 30:
        level = "Difficult"
    else:
        level = "Very Difficult"

    return int(fre), {
        "flesch_score": round(fre, 1),
        "reading_level": level,
        "avg_sentence_length": round(avg_sent_len, 1),
        "avg_syllables_per_word": round(avg_syllables, 2),
        "total_words": num_words,
        "total_sentences": num_sentences
    }


# 6. Consistency Check
def check_consistency(text):
    issues = []
    score = 100

    words = tokenize(text)
    word_counts = Counter(words)

    # Overused words (excluding common stopwords)
    stopwords = {'the','a','an','is','are','was','were','be','been','being',
                 'have','has','had','do','does','did','will','would','could',
                 'should','may','might','must','shall','can','need','dare',
                 'ought','used','to','of','in','for','on','with','at','by',
                 'from','up','about','into','through','during','before',
                 'after','above','below','between','out','off','over',
                 'under','again','further','then','once','and','but','or',
                 'nor','so','yet','both','either','neither','not','only',
                 'own','same','than','too','very','just','that','this',
                 'it','its','i','you','he','she','we','they','what','which',
                 'who','whom','if','as','because','while','although','though'}

    overused = [(w, c) for w, c in word_counts.items()
                if c > 4 and w not in stopwords and len(w) > 3]
    overused.sort(key=lambda x: -x[1])

    if overused:
        penalty = min(25, len(overused) * 5)
        score -= penalty
        top = [f'"{w}"({c}x)' for w, c in overused[:4]]
        issues.append(f"Overused words: {', '.join(top)}")

    # Contradictions (simple check)
    lower = text.lower()
    if ('always' in lower and 'never' in lower) or \
       ('is' in lower and 'is not' in lower and lower.index('is') < lower.index('is not') - 5):
        issues.append("Possible contradiction detected")
        score -= 15

    # Number consistency (mixing formats)
    written_nums = re.findall(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', lower)
    digit_nums = re.findall(r'\b[1-9]\b', text)
    if written_nums and digit_nums and len(written_nums) > 1 and len(digit_nums) > 1:
        issues.append("Inconsistent number formats (mixing words and digits)")
        score -= 8

    return max(0, score), issues


# ──────────────────────────────────────────────
#  MAIN ANALYZER
# ──────────────────────────────────────────────

def analyze_response(response_text, expected_keywords="", expected_response=""):
    results = {}

    g_score, g_issues = check_grammar(response_text)
    c_score, c_issues = check_clarity(response_text)
    t_score, t_details = check_tone(response_text)
    r_score, r_details = check_relevance(response_text, expected_keywords)
    rd_score, rd_details = check_readability(response_text)
    co_score, co_issues = check_consistency(response_text)

    # Weighted overall score
    overall = int(
        g_score  * 0.20 +
        c_score  * 0.25 +
        t_score  * 0.15 +
        r_score  * 0.20 +
        rd_score * 0.10 +
        co_score * 0.10
    )

    if overall >= 85:
        grade = "A"
        verdict = "Excellent Response"
    elif overall >= 70:
        grade = "B"
        verdict = "Good Response"
    elif overall >= 55:
        grade = "C"
        verdict = "Average Response"
    elif overall >= 40:
        grade = "D"
        verdict = "Below Average"
    else:
        grade = "F"
        verdict = "Poor Response"

    results = {
        "overall_score": overall,
        "grade": grade,
        "verdict": verdict,
        "breakdown": {
            "grammar":      {"score": g_score,  "issues": g_issues,   "weight": "20%"},
            "clarity":      {"score": c_score,  "issues": c_issues,   "weight": "25%"},
            "tone":         {"score": t_score,  "details": t_details, "weight": "15%"},
            "relevance":    {"score": r_score,  "details": r_details, "weight": "20%"},
            "readability":  {"score": rd_score, "details": rd_details,"weight": "10%"},
            "consistency":  {"score": co_score, "issues": co_issues,  "weight": "10%"},
        }
    }
    return results


def compare_responses(expected, generated, keywords=""):
    exp_result = analyze_response(expected, keywords)
    gen_result = analyze_response(generated, keywords)

    diff = gen_result["overall_score"] - exp_result["overall_score"]

    return {
        "expected_analysis": exp_result,
        "generated_analysis": gen_result,
        "score_difference": diff,
        "comparison": "Generated is better" if diff > 0 else
                      "Expected is better" if diff < 0 else "Equal quality"
    }


def print_report(result, title="ANALYSIS REPORT"):
    SEP = "═" * 60
    sep = "─" * 60

    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    print(f"  Overall Score : {result['overall_score']}/100  |  Grade: {result['grade']}  |  {result['verdict']}")
    print(sep)

    for metric, data in result["breakdown"].items():
        score = data["score"]
        bar = "█" * (score // 5) + "░" * (20 - score // 5)
        print(f"\n  {metric.upper():15} [{bar}] {score}/100  (weight: {data['weight']})")

        if "issues" in data and data["issues"]:
            for issue in data["issues"]:
                print(f"    ⚠  {issue}")
        elif "issues" in data:
            print(f"    ✓  No issues found")

        if "details" in data:
            d = data["details"]
            if metric == "tone":
                print(f"    → Tone: {d.get('tone','N/A')} | Positive signals: {d.get('positive_signals',0)} | Uncertainty phrases: {d.get('uncertainty_phrases',0)}")
            elif metric == "readability":
                print(f"    → Level: {d.get('reading_level','N/A')} | Words: {d.get('total_words',0)} | Avg sentence: {d.get('avg_sentence_length',0)} words")
            elif metric == "relevance":
                print(f"    → Coverage: {d.get('coverage','N/A')} | Matched: {d.get('matched',[])} | Missing: {d.get('missing',[])}")

    print(f"\n{SEP}\n")


# ──────────────────────────────────────────────
#  DEMO USAGE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI RESPONSE QUALITY CHECKER — by Karan Bisht")
    print("="*60)

    # --- DEMO 1: Single response analysis ---
    sample_response = """
    Python is basically a very popular programming language that is used
    for various things like web development, data science, and AI stuff.
    It was created by Guido van Rossum and was released in 1991. Python's
    syntax is quite simple and it can actually be learned easily by beginners.
    The language has lots of libraries and frameworks that make development faster.
    Machine learning models are often trained using Python because it has
    excellent libraries like TensorFlow and PyTorch. Python is definitely
    one of the best languages for beginners and experts alike.
    """

    print("\n[DEMO 1] Analyzing a sample AI response...\n")
    result = analyze_response(sample_response, expected_keywords="python, programming, syntax, libraries")
    print_report(result, "SINGLE RESPONSE ANALYSIS")

    # --- DEMO 2: Compare expected vs generated ---
    expected = """
    Python is a high-level, interpreted programming language known for its
    clear syntax and readability. Created by Guido van Rossum in 1991, it
    supports multiple programming paradigms. Python's extensive standard
    library and active community make it ideal for web development, data
    analysis, and machine learning tasks.
    """

    generated = """
    Python is literally just a coding language. It's very popular and
    basically everyone uses it. You can do web stuff and AI stuff with it.
    It's quite easy to learn and has lots of things you can do with it.
    I think it was made by someone a long time ago. It's definitely good.
    """

    print("\n[DEMO 2] Comparing expected vs generated responses...\n")
    comparison = compare_responses(expected, generated, keywords="python, programming, syntax, library")
    print_report(comparison["expected_analysis"], "EXPECTED RESPONSE")
    print_report(comparison["generated_analysis"], "GENERATED RESPONSE")
    print(f"  COMPARISON RESULT: {comparison['comparison']}")
    print(f"  Score Difference : {comparison['score_difference']:+d} points\n")

    # --- DEMO 3: High quality response ---
    high_quality = """
    Python is a versatile, high-level programming language designed for
    readability and simplicity. Guido van Rossum introduced it in 1991,
    emphasizing clean syntax that reduces the cost of program maintenance.
    The language supports object-oriented, procedural, and functional
    programming paradigms. Its comprehensive standard library, combined
    with third-party packages such as NumPy, Pandas, and TensorFlow,
    makes Python a leading choice for data science, web development,
    and artificial intelligence applications.
    """

    print("\n[DEMO 3] Analyzing a high-quality response...\n")
    result3 = analyze_response(high_quality, expected_keywords="python, programming, syntax, libraries, data science")
    print_report(result3, "HIGH QUALITY RESPONSE ANALYSIS")

    print("  ✓ Analysis complete. Export results with json.dumps(result)")
    print("  ✓ Call analyze_response(text, keywords) for custom analysis")
    print("  ✓ Call compare_responses(expected, generated) for comparison\n")
