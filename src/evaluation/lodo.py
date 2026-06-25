def lodo_splits(datasets):
    names = list(datasets.keys())
    for held_out in names:
        train_names = [name for name in names if name != held_out]
        yield train_names, held_out


def run_lodo_evaluation(model_fn, datasets):
    results = {}
    for train_names, held_out in lodo_splits(datasets):
        train_sets = [datasets[name] for name in train_names]
        test_set = datasets[held_out]
        model = model_fn()
        results[held_out] = {
            "train_sets": train_names,
            "test_set": held_out,
            "metrics": {},
        }
        _ = train_sets, test_set, model
    return results
