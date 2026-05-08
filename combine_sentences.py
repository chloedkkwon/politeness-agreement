import csv

def get_subject_marker(subject):
    """Extract the subject particle from the subject string."""
    if subject.endswith("께서"):
        return "께서"
    elif subject.endswith("이"):
        return "이"
    elif subject.endswith("가"):
        return "가"
    else:
        return None  # fallback
    
# ===== Experiment 1 =====

subj_1_honorable = [
    "선생님이", "교수님이",
    "어머니가", "아버지가",
    "할머니가", "할아버지가",
    "사장님이", "팀장님이",
    "이모님이", "손님이"
    ]

subj_1_plain = [
    "아이가" , "꼬마가",
    "남자가", "여자가",
    "딸이", "아들이",
     "친구가", "막내가",
     "초등학생이", "여동생이"
     ]

vp_1_honorific = [
    "책을 읽으셨다",
    "파스타를 드셨다",
    "호통을 치셨다",
    "핸드폰을 사셨다",
    "편지를 쓰셨다",
    "노래를 부르셨다",
    "티비를 보셨다",
    "산책을 하셨다",
    "정원을 가꾸셨다",
    "빨래를 널으셨다"]

vp_1_plain = [
    "책을 읽었다",
    "파스타를 먹었다",
    "호통을 쳤다",
    "핸드폰을 샀다",
    "편지를 썼다", 
    "노래를 불렀다",
    "티비를 봤다",
    "산책을 했다",
    "정원을 가꿨다",
    "빨래를 널었다"
]

adv = "어제 저녁에 혼자서"

subjects = subj_1_honorable + subj_1_plain
verb_phrases = list(zip(vp_1_honorific, vp_1_plain))

templates = {
    "close": "{} {} {}.", # adv at beginning
    "far": "{1} {0} {2}." # adv in middle
}

test_data_1 = []
exp_idx = 1
item_number = 1 
for distance, template in templates.items():
    for subject in subjects:
        if subject in subj_1_honorable:
            subject_type = "honorableS"
        else:
            subject_type = "plainS"
        
        for verb_phrase in verb_phrases:
            honorific_verb, plain_verb = verb_phrase
        
            sentence = template.format(adv, subject, honorific_verb)
            test_data_1.append({
                "exp_idx": exp_idx, 
                "item_number": item_number, 
                "condition": subject_type + "-" + "honorificV",
                "sentence": sentence,
                "distance": distance,
                "subject": subject, 
                "subject_marker": get_subject_marker(subject),
                "verb_phrase": honorific_verb,
                "target_phrase": honorific_verb.split()[1],
                "match": subject_type[:3] == "hon",
                "grammatical": None
            })
            
            sentence = template.format(adv, subject, plain_verb)
            test_data_1.append({
                "exp_idx": exp_idx, 
                "item_number": item_number, 
                "condition": subject_type + "-" + "plainV",
                "sentence": sentence,
                "distance": distance,
                "subject": subject, 
                "subject_marker": get_subject_marker(subject),
                "verb_phrase": plain_verb,
                "target_phrase": plain_verb.split()[1],
                "match": subject_type[:3] == "pla",
                "grammatical": None
            })

            item_number += 1


# ===== Experiment 2 =====
subj_2 = [
    ("선생님이", "선생님께서"), ("교수님이", "교수님께서"),
    ("어머니가", "어머니께서"), ("아버지가", "아버지께서"),
    ("할머니가", "할머니께서"), ("할아버지가", "할아버지께서"),
    ("사장님이", "사장님께서"), ("팀장님이", "팀장님께서"),
    ("이모님이", "이모님께서"), ("손님이", "손님께서")
]

vp_2 = [
    ("책을 읽었다", "책을 읽으셨다"), ("파스타를 먹었다", "파스타를 드셨다"),
    ("호통을 쳤다", "호통을 치셨다"), ("핸드폰을 샀다", "핸드폰을 사셨다"),
    ("편지를 썼다", "편지를 쓰셨다"), ("노래를 불렀다", "노래를 부르셨다"),
    ("티비를 봤다", "티비를 보셨다"), ("산책을 했다", "산책을 하셨다"),
    ("정원을 가꿨다", "정원을 가꾸셨다"), ("빨래를 널었다", "빨래를 널으셨다")
]


def generate_sentence_dict(exp_idx, subjects, verb_phrases, adverb):
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
    
    item_number = 1  # initialize item number

    for distance, template in templates.items():
        for subj_pair in subjects:
            plain_subj, hon_subj = subj_pair

            for vp_pair in verb_phrases:
                plain_vp, hon_vp = vp_pair

                # Condition 1: No honorific
                sentence = template.format(adverb, plain_subj, plain_vp)
                test_data.append({
                    "exp_idx": exp_idx,
                    "item_number": item_number,
                    "condition": "no_honorific",
                    "sentence": sentence,
                    "distance": distance,
                    "subject": plain_subj, 
                    "subject_marker": get_subject_marker(plain_subj),
                    "verb_phrase": plain_vp,
                    "target_phrase": plain_vp.split()[1],
                    "match": True,
                    "grammatical": True
                })

                # Condition 2: Verb only
                sentence = template.format(adverb, plain_subj, hon_vp)
                test_data.append({
                    "exp_idx": exp_idx,
                    "item_number": item_number,
                    "condition": "verb_only",
                    "sentence": sentence,
                    "distance": distance,
                    "subject": plain_subj,
                    "subject_marker": get_subject_marker(plain_subj),
                    "verb_phrase": hon_vp,
                    "target_phrase": hon_vp.split()[1],
                    "match": False,
                    "grammatical": True
                })
                
                # Condition 3: Noun only
                sentence = template.format(adverb, hon_subj, plain_vp)
                test_data.append({
                    "exp_idx": exp_idx,
                    "item_number": item_number,
                    "condition": "noun_only",
                    "sentence": sentence,
                    "distance": distance,
                    "subject": hon_subj,
                    "subject_marker": get_subject_marker(hon_subj),
                    "verb_phrase": plain_vp,
                    "target_phrase": plain_vp.split()[1],
                    "match": False,
                    "grammatical": False
                })
                
                # Condition 4: All honorific
                sentence = template.format(adverb, hon_subj, hon_vp)
                test_data.append({
                    "exp_idx": exp_idx,
                    "item_number": item_number,
                    "condition": "all_honorific",
                    "sentence": sentence,
                    "distance": distance,
                    "subject": hon_subj,
                    "subject_marker": get_subject_marker(hon_subj),
                    "verb_phrase": hon_vp,
                    "target_phrase": hon_vp.split()[1],
                    "match": True,
                    "grammatical": True
                })

                item_number += 1

    return test_data
test_data_2 = generate_sentence_dict(2, subj_2, vp_2, adv)

# ===== Experiment 3 =====
subj_3 = [
    ("아이가", "아이께서"),
    ("꼬마가", "꼬마께서"),
    ("남자가", "남자께서"),
    ("여자가", "여자께서"),
    ("딸이", "딸께서"),
    ("아들이", "아들께서"),
    ("친구가", "친구께서"),
    ("막내가", "막내께서"),
    ("초등학생이", "초등학생께서"),
    ("여동생이", "여동생께서")
]

vp_3 = [
    ("책을 읽었다", "책을 읽으셨다"),
    ("파스타를 먹었다", "파스타를 드셨다"),
    ("호통을 쳤다", "호통을 치셨다"),
    ("핸드폰을 샀다", "핸드폰을 사셨다"),
    ("편지를 썼다", "편지를 쓰셨다"),
    ("노래를 불렀다", "노래를 부르셨다"),
    ("티비를 봤다", "티비를 보셨다"),
    ("산책을 했다", "산책을 하셨다"),
    ("정원을 가꿨다", "정원을 가꾸셨다"),
    ("빨래를 널었다", "빨래를 널으셨다")
]

test_data_3 = generate_sentence_dict(3, subj_3, vp_3, adv)


test_data = test_data_1 + test_data_2 + test_data_3
filename = "data/sentences_allExp.csv"
fieldnames = list(test_data[0].keys())
with open(filename, 'w', encoding='utf-8-sig', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test_data)