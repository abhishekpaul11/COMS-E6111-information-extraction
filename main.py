import sys

import spacy

from extract_spanbert import extract_spanbert, SPANBERT_RELATIONS
from google_search import google_search
from scrape_text import extract_text
from spanbert import SpanBERT

nlp = spacy.load("en_core_web_lg")


def process_page(text, r, method, threshold=None):
    print(f'\n\tAnnotating the webpage using spacy...')
    doc = nlp(text)

    print(f'\tExtracted {len(list(doc.sents))} sentence(s). Processing each sentence one by one to for presence of right pair of named'
          f' entity types; if so, will run the second pipeline...')

    tuples_from_doc, total_rel, sentences_of_interest = extract_spanbert(doc, r, threshold, spanbert_model)

    print(f'\n\tExtracted annotations for {sentences_of_interest} out of total {len(list(doc.sents))} sentence(s)')
    print(f'\tRelations extracted from this website: {len(tuples_from_doc)} (Overall: {total_rel})')
    return tuples_from_doc

    # if method == "-spanbert":
    #     tuples = extract_spanbert(valid_sentences, r, threshold, spanbert_model)
    # else:
    #     for sent in valid_sentences:
    #         new_tuples = extract_with_gemini(sent.text, r)
    #         tuples.extend(new_tuples)
    #
    # return tuples


if len(sys.argv) != 6:
    print(
        "Usage: python3 project2.py [-spanbert|-gemini] <r> <t> <q> <k>")
    sys.exit(1)

method = sys.argv[1]
google_api_key = 'AIzaSyAoO2ShS96E0SUS8gs7r4q907iZL8fzEQ4'
google_engine_id = '01fe29cf6df6d48b2'
gemini_api_key = ''
r = int(sys.argv[2])
t = float(sys.argv[3])
q = sys.argv[4]
k = int(sys.argv[5])

if method not in ["-spanbert", "-gemini"] or r not in [1, 2, 3, 4] or k <= 0 or type(k) != int or (
        method == "-spanbert" and not 0 <= t <= 1):
    print("Invalid arguments.")
    sys.exit(1)


if method == "-spanbert":
    spanbert_model = SpanBERT("./pretrained_spanbert")
# else:
#     gemini_api = GeminiAPI(gemini_api_key)


print("____")
print(f"Parameters:\n")
print(f"Client key\t= {google_api_key}")
print(f"Engine key\t= {google_engine_id}")
print(f"Gemini key\t= {gemini_api_key}")
print(f"Method\t\t= {method[1:]}")
print(f"Relation\t= {['Schools_Attended', 'Work_For', 'Live_In', 'Top_Member_Employees'][r-1]}")
print(f"Threshold\t= {t}")
print(f"Query\t\t= {q}")
print(f"# of Tuples\t= {k}")
print("\nProcessing your results. This may take a while...")


X = set()  # (subject, object, confidence) tuples
processed_urls = set()
processed_queries = set()

current_query = q
iteration = 0
stalled = False
while len(X) < k:
    print(f"\n=========== Iteration: {iteration} - Query: {current_query} ===========")
    results = google_search(current_query, google_api_key, google_engine_id)

    url_index = 1
    for result in results.get('items', []):
        if 'link' not in result:
            continue

        url = result['link']

        if url in processed_urls:
            continue
        processed_urls.add(url)

        print(f"\nURL ( {url_index} / 10): {url}")
        url_index += 1

        print('\n\tFetching text from url...')

        if 'fileFormat' in result:
            print('\tUnable to extract text from URL (non-html file). Continuing.')
            continue

        text = extract_text(url)
        if not text:
            print('\tUnable to extract text from URL. Continuing.')
            continue

        new_tuples = process_page(text, r, method, t if method == "-spanbert" else None)
        X.update(new_tuples)

    X_dict = {}
    for subj, obj, conf in X:
        key = (subj, obj)
        if key not in X_dict or (method == "-spanbert" and conf > X_dict[key][2]):
            X_dict[key] = (subj, obj, conf)
    X = set(X_dict.values())

    if method == "-spanbert":
        result = sorted(X, key=lambda x: x[2], reverse=True)[:k]
    else:
        result = list(X)[:k]

    unused = [(subj, obj, conf) for (subj, obj, conf) in X if f"{subj} {obj}" not in processed_queries]
    if unused:
        next_tuple = max(unused, key=lambda x: x[2]) if method == "-spanbert" else unused[0]
        current_query = f"{next_tuple[0]} {next_tuple[1]}"
        processed_queries.add(current_query)
    else:
        print("\nStalled: No unused tuples for new query. Returning the ones extracted so far.")
        stalled = True

    relation_name = SPANBERT_RELATIONS[r] if method == "-spanbert" else {1: "Schools_Attended", 2: "Work_For", 3: "Live_In", 4: "Top_Member_Employees"}[r]
    num_tuples = f'Top {k} out of {len(X)} extracted' if len(X) > k else len(X)

    print(f"\n================== ALL RELATIONS for {relation_name} ( {num_tuples} ) ==================")
    for subj, obj, conf in result:
        print(f"Confidence: {conf}\t\t| Subject: {subj}\t\t| Object: {obj}")

    iteration += 1

    if stalled:
        break

print(f'\nTotal # of iterations = {iteration}')