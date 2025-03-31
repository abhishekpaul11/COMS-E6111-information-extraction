import time
import google.generativeai as genai

from spacy_help_functions import has_entities_of_interest


def get_gemini_completion(gemini_model, prompt, max_tokens=200, temperature=0.2, top_p=1, top_k=32):
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    response = gemini_model.generate_content(prompt, generation_config=generation_config)
    return response.text.strip() if response.text else None


def extract_with_gemini(doc, r, subject_type, object_type, gemini_model):
    examples = {
        1: {"sentence": "Jeff Bezos graduated from Princeton University in 1986.",
            "relation": '[("Jeff Bezos", "Princeton University")]'},
        2: {"sentence": "Alec Radford works as a researcher at OpenAI.", "relation": '[("Alec Radford", "OpenAI")]'},
        3: {"sentence": "Mariah Carey resides in New York City with her family.",
            "relation": '[("Mariah Carey", "New York City")]'},
        4: {"sentence": "Jensen Huang is the CEO of Nvidia.", "relation": '[("Nvidia", "Jensen Huang")]'}
    }
    relation_map = {1: "Schools_Attended", 2: "Work_For", 3: "Live_In", 4: "Top_Member_Employees"}
    target_relation = relation_map[r]
    example = examples[r]

    tuples = []
    num_relations = 0
    sentence_index = 1
    sentences_extracted_from = 0

    for sent in doc.sents:
        if has_entities_of_interest(sent, subject_type, object_type):
            prompt = f"""
            You are tasked with extracting {target_relation} relations from text. 
            Here is an example to guide you:
            - Sentence: "{example['sentence']}"
            - Extracted relation: {example['relation']}
        
            Now, extract {target_relation} relations from the following sentence. 
            Return the result as a list of tuples in the format: [("subject", "object"), ...].
            If no relations are found, return an empty list.
            Sentence: "{sent.text}"
            """

            try:
                time.sleep(5)
                response = get_gemini_completion(gemini_model, prompt)
                start = response.index('[')
                end = response.rindex(']') + 1
                response =  response[start:end]

                if response:
                    extracted = eval(response)

                    if len(extracted) > 0:
                        sentences_extracted_from += 1

                    for subj, obj in extracted:
                        print("\n\t\t=== Extracted Relation ===")
                        print("\t\tSentence: {}".format(sent.text))
                        print("\t\tSubject: {} ; Object: {}".format(subj, obj))

                        if (subj, obj, 1.0) in tuples:
                            print("\t\tDuplicate. Ignoring this.")
                        else:
                            tuples.append((subj, obj, 1.0))
                            print("\t\tAdding to set of extracted relations.")
                        print("\t\t==========")

                        num_relations += 1

            except Exception as e:
                print(f"\n\t\tGemini error: {e}")

        if sentence_index % 5 == 0:
            print(f'\n\tProcessed {sentence_index} / {len(list(doc.sents))} sentences')
        sentence_index += 1

    return tuples, num_relations, sentences_extracted_from