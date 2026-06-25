def evaluate_by_center(dataset, center_key="center"):
    by_center = {}
    for record in dataset:
        center = record.get(center_key, "unknown")
        by_center.setdefault(center, []).append(record)
    return {center: {} for center in by_center}
