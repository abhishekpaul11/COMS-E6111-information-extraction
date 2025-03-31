from spacy_help_functions import extract_relations

SPANBERT_RELATIONS = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}


def extract_spanbert(doc, r, subject_type, object_type, threshold, spanbert_model):
    target_relation = SPANBERT_RELATIONS[r]
    entities_of_interest = {subject_type, *(object_type if isinstance(object_type, set) else [object_type])}

    relations, total_rel_count, sentences_extracted_from \
        = extract_relations(doc, spanbert_model, target_relation, subject_type, object_type, entities_of_interest, conf=threshold)

    tuples = []
    for (subj, rel, obj), conf in relations.items():
        tuples.append((subj, obj, conf))

    return tuples, total_rel_count, sentences_extracted_from
