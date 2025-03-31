def extract_with_gemini(sentence, r):
    """Extract relations using Gemini API with one-shot learning prompt."""
    examples = {
        1: {
            "sentence": "Jeff Bezos graduated from Princeton University in 1986.",
            "relation": '["Jeff Bezos", "Schools_Attended", "Princeton University"]'
        },
        2: {
            "sentence": "Alec Radford works as a researcher at OpenAI.",
            "relation": '["Alec Radford", "Work_For", "OpenAI"]'
        },
        3: {
            "sentence": "Mariah Carey resides in New York City with her family.",
            "relation": '["Mariah Carey", "Live_In", "New York City"]'
        },
        4: {
            "sentence": "Jensen Huang is the CEO of Nvidia.",
            "relation": '["Nvidia", "Top_Member_Employees", "Jensen Huang"]'
        }
    }

    relation_map = {1: "Schools_Attended", 2: "Work_For", 3: "Live_In", 4: "Top_Member_Employees"}
    target_relation = relation_map[r]
    example = examples[r]

    prompt = f"""
    You are tasked with extracting {target_relation} relations from text. 
    Here is an example to guide you:
    - Sentence: "{example['sentence']}"
    - Extracted relation: {example['relation']}

    Now, extract {target_relation} relations from the following sentence. 
    Return the result as a list of tuples in the format: [(subject, object), ...].
    If no relations are found, return an empty list.
    Sentence: "{sentence}"
    """

    try:
        time.sleep(5)  # Rate limit precaution
        # Replace with actual Gemini API call: response = gemini_api.generate(prompt)
        response = "[('dummy', 'dummy')]"  # Dummy response for demo
        extracted = eval(response)  # Adjust parsing based on actual API output
        return [(subj, obj, 1.0) for (subj, obj) in extracted]
    except Exception as e:
        print(f"Gemini error: {e}")
        return []