# Korean honorific test data generator
import csv


subjects = [
    ("선생님이", "선생님께서"), ("교수님이", "교수님께서"),
    ("어머니가", "어머니께서"), ("아버지가", "아버지께서"),
    ("할머니가", "할머니께서"), ("할아버지가", "할아버지께서"),
    ("사장님이", "사장님께서"), ("팀장님이", "팀장님께서"),
    ("이모님이", "이모님께서"), ("손님이", "손님께서")
]

verb_phrases = [
    ("책을 읽었다", "책을 읽으셨다"), ("파스타를 먹었다", "파스타를 드셨다"),
    ("호통을 쳤다", "호통을 치셨다"), ("핸드폰을 샀다", "핸드폰을 사셨다"),
    ("편지를 썼다", "편지를 쓰셨다"), ("노래를 불렀다", "노래를 부르셨다"),
    ("티비를 봤다", "티비를 보셨다"), ("산책을 했다", "산책을 하셨다"),
    ("정원을 가꿨다", "정원을 가꾸셨다"), ("빨래를 널었다", "빨래를 널으셨다")
]

adverb = "어제 저녁에 혼자서"

def generate_sentence_dict():
    """
    Generate test data with 8 total conditions:
    - 4 honorific conditions (no_honorific, verb_only, noun_only, all_honoric)
    - 2 distance conditions (close, far)
        = 4 x 2 = 8 combinations
    """
    test_data = []

    templates = {
        "close": "{} {} {}.", # adv at beginning
        "far": "{1} {0} {2}." # adv in middle
    }

    for distance, template in templates.items():
        for subj_pair in subjects:
            plain_subj, hon_subj = subj_pair

            for vp_pair in verb_phrases:
                plain_vp, hon_vp = vp_pair

                # Condition 1: No honorific
                sentence = template.format(adverb, plain_subj, plain_vp)
                test_data.append({
                    "condition": "no_honorific",
                    "distance": distance,
                    "sentence": sentence,
                    "subject": plain_subj, 
                    "verb_phrase": plain_vp,
                    "target_phrase": plain_vp,
                    "grammatical": True,
                    "mask_char_pos": len(sentence) - 3
                })

                # Condition 2: Verb only
                sentence = template.format(adverb, plain_subj, hon_vp)
                test_data.append({
                    "condition": "verb_only",
                    "distance": distance,
                    "sentence": sentence,
                    "subject": plain_subj,
                    "verb_phrase": hon_vp,
                    "target_phrase": hon_vp,
                    "grammatical": False,
                    "mask_char_pos": len(sentence) - 3
                })
                
                # Condition 3: Noun only
                sentence = template.format(adverb, hon_subj, plain_vp)
                test_data.append({
                    "condition": "noun_only",
                    "distance": distance,
                    "sentence": sentence,
                    "subject": hon_subj,
                    "verb_phrase": plain_vp,
                    "target_phrase": plain_vp,
                    "grammatical": True,
                    "mask_char_pos": len(sentence) - 3
                })
                
                # Condition 4: All honorific
                sentence = template.format(adverb, hon_subj, hon_vp)
                test_data.append({
                    "condition": "all_honorific",
                    "distance": distance,
                    "sentence": sentence,
                    "subject": hon_subj,
                    "verb_phrase": hon_vp,
                    "target_phrase": hon_vp,
                    "grammatical": True,
                    "mask_char_pos": len(sentence) - 2
                })
    return test_data

test_data = generate_sentence_dict()

filename = "data/sentences.csv"
fieldnames = list(test_data[0].keys())
with open(filename, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test_data)
