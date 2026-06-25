def evaluate_by_culture(dataset, culture_key="culture"):
    by_culture = {}
    for record in dataset:
        culture = record.get(culture_key, "unknown")
        by_culture.setdefault(culture, []).append(record)
    return {culture: {} for culture in by_culture}
