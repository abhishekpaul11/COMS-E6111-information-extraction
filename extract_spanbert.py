from spacy_help_functions import extract_relations


RELATION_ENTITY_TYPES = {
    1: ("PERSON", "ORGANIZATION"),  # Schools_Attended
    2: ("PERSON", "ORGANIZATION"),  # Work_For
    3: ("PERSON", {"LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"}),  # Live_In
    4: ("ORGANIZATION", "PERSON")  # Top_Member_Employees
}

SPANBERT_RELATIONS = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}


def extract_spanbert(doc, r, threshold, spanbert_model):
    target_relation = SPANBERT_RELATIONS[r]
    subject_type, object_type = RELATION_ENTITY_TYPES[r]
    entities_of_interest = {subject_type, *(object_type if isinstance(object_type, set) else [object_type])}

    relations, total_rel_count, sentences_extracted_from \
        = extract_relations(doc, spanbert_model, target_relation, subject_type, object_type, entities_of_interest, conf=threshold)

    tuples = []
    for (subj, rel, obj), conf in relations.items():
        tuples.append((subj, obj, conf))

    return tuples, total_rel_count, sentences_extracted_from
