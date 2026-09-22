from collections import defaultdict
from tqdm import tqdm

from .spacy_utils import upos_dict, dep_dict
from sklearn.metrics import classification_report


flip_upos_dict = {v: k for k, v in upos_dict.items()}
flip_dep_dict = {v: k for k, v in dep_dict["en_core_web_sm"].items()}

def error_analysis_cue(tokens, pos_ids, dep_ids, y_true, y_pred):
    error_analysis_cue_pos(tokens, pos_ids, y_true, y_pred)
    error_analysis_cue_dep(tokens, dep_ids, y_true, y_pred)

def error_analysis_cue_pos(tokens, pos_ids, y_true, y_pred):
    assert len(tokens) == len(pos_ids) == len(y_true) == len(y_pred), print(len(tokens), len(pos_ids), len(y_true), len(y_pred))
    new_ypred, new_ytrue = [], []
    for i in range(len(tokens)):
        new_ypred.append(f"{flip_upos_dict[pos_ids[i]]}_{y_pred[i]}")
        new_ytrue.append(f"{flip_upos_dict[pos_ids[i]]}_{y_true[i]}")
    report = classification_report(new_ytrue, new_ypred)
    print(report)
    report_dict = classification_report(new_ytrue, new_ypred, output_dict=True)
    print(report_dict)
    impact_results = find_classes_biggest_negative_impact(report_dict)
    # Sort by contribution ascending (worst contributors first)
    worst_impact = sorted(impact_results.items(), key=lambda x: x[1]["f1"])

    # Print
    for label, imp in worst_impact:
        print(
            f"{label:10s} | F1: {report_dict[label]['f1-score']:.2f} | Support: {report_dict[label]['support']} | Contribution: {imp['delta_if_removed']:.5f}")

def error_analysis_cue_dep(tokens, dep_ids, y_true, y_pred):
    assert len(tokens) == len(dep_ids) == len(y_true) == len(y_pred), print(len(tokens), len(dep_ids), len(y_true), len(y_pred))
    new_ypred, new_ytrue = [], []
    for i in range(len(tokens)):
        new_ypred.append(f"{flip_dep_dict[dep_ids[i]]}_{y_pred[i]}")
        new_ytrue.append(f"{flip_dep_dict[dep_ids[i]]}_{y_true[i]}")
    report = classification_report(new_ytrue, new_ypred)
    print(report)
    report_dict = classification_report(new_ytrue, new_ypred, output_dict=True)
    print(report_dict)
    impact_results = find_classes_biggest_negative_impact(report_dict)
    # Sort by contribution ascending (worst contributors first)
    worst_impact = sorted(impact_results.items(), key=lambda x: x[1]["f1"])

    # Print
    for label, imp in worst_impact:
        print(
            f"{label:10s} | F1: {report_dict[label]['f1-score']:.2f} | Support: {report_dict[label]['support']} | Contribution: {imp['delta_if_removed']:.5f}")


def find_classes_biggest_negative_impact(report_dict):
    impact_results = dict()
    for label, metrics in report_dict.items():
        if label not in ["accuracy", "macro avg", "f1-score", "weighted avg"]:
            support = metrics["support"]
            f1 = metrics["f1-score"]

            if support == 0:
                continue  # skip classes with 0 support

            old_weighted_f1_entangled = report_dict["weighted avg"]["f1-score"] * report_dict["weighted avg"]["support"]
            new_support = report_dict["weighted avg"]["support"] - support
            new_weighted_f1_sum = old_weighted_f1_entangled - (support * f1)
            new_weighted_f1 = new_weighted_f1_sum / new_support

            delta = new_weighted_f1 - report_dict["weighted avg"]["f1-score"]
            impact_results[label] = {
                "support": support,
                "f1": f1,
                "delta_if_removed": delta
            }
    return impact_results


def error_analysis_scope(tokens, pos_ids, dep_ids, dep_indices, y_true, y_pred, cue_per_sent):
    # print(dep_indices)
    # print(type(dep_indices))
    depths = []
    for idx in tqdm(range(len(dep_indices)), desc="Extracting depths"):
        tree = TreeConstructor.calculate_node_depths(dep_indices[idx])
        for jdx in range(len(tokens[idx])):
            try:
                depths.append(tree[jdx])
            except KeyError:
                depths.append(-1)

    print(len(tokens), len(dep_indices), len(pos_ids), len(y_true), len(y_pred), len(dep_ids), len(depths), len(cue_per_sent))
    print(set(depths))
    print(cue_per_sent)
    new_ypred, new_ytrue = [], []
    for i in range(len(pos_ids)):
        new_ypred.append(f"{depths[i]}_{y_pred[i]}")
        new_ytrue.append(f"{depths[i]}_{y_true[i]}")
    report = classification_report(new_ytrue, new_ypred)
    print(report)
    report_dict = classification_report(new_ytrue, new_ypred, output_dict=True)
    print(report_dict)
    impact_results = find_classes_biggest_negative_impact(report_dict)
    # Sort by contribution ascending (worst contributors first)
    worst_impact = sorted(impact_results.items(), key=lambda x: x[1]["f1"])

    # Print
    for label, imp in worst_impact:
        print(
            f"{label:10s} | F1: {report_dict[label]['f1-score']:.2f} | Support: {report_dict[label]['support']} | Contribution: {imp['delta_if_removed']:.5f}")

    print_example = False
    i = 0
    print(sum([len(sub) for sub in tokens]), len(new_ypred))
    if print_example:
        for idx in range(len(tokens)):
            is_example = False
            start = i
            for jdx in range(len(tokens[idx])):
                if new_ytrue[i] == "0_1":
                    is_example = True
                i += 1
            end = i
            if is_example:
                print(tokens[idx])
                print(dep_indices[idx])
                print(new_ypred[start:end])
                print(new_ytrue[start:end])


def error_analysis_scope_numcues(tokens, pos_ids, dep_ids, dep_indices, y_true, y_pred, cue_per_sent):

    new_ypred, new_ytrue = [], []
    for i in range(len(pos_ids)):
        new_ypred.append(f"{cue_per_sent[i]}_{y_pred[i]}")
        new_ytrue.append(f"{cue_per_sent[i]}_{y_true[i]}")
    report = classification_report(new_ytrue, new_ypred)
    print(report)
    report_dict = classification_report(new_ytrue, new_ypred, output_dict=True)
    print(report_dict)
    impact_results = find_classes_biggest_negative_impact(report_dict)
    # Sort by contribution ascending (worst contributors first)
    worst_impact = sorted(impact_results.items(), key=lambda x: x[1]["f1"])

    # Print
    rows = []
    pos_drop, neg_drop = [0, 0], [0, 0]
    for label, imp in worst_impact:
        row = f"{label:10s} | F1: {report_dict[label]['f1-score']:.2f} | Support: {report_dict[label]['support']} | Contribution: {imp['delta_if_removed']:.5f}"
        print(row)
        # print(f"{label}")
        rows.append(row)
        if f"{label}" == "2_1":
            pos_drop[1] = report_dict[label]['f1-score']
        elif f"{label}" == "1_1":
            pos_drop[0] = report_dict[label]['f1-score']
        elif f"{label}" == "2_0":
            neg_drop[1] = report_dict[label]['f1-score']
        elif f"{label}" == "1_0":
            neg_drop[0] = report_dict[label]['f1-score']

    print_example = False
    i = 0
    print(sum([len(sub) for sub in tokens]), len(new_ypred))
    if print_example:
        for idx in range(len(tokens)):
            is_example = False
            start = i
            for jdx in range(len(tokens[idx])):
                if new_ytrue[i] == "0_1":
                    is_example = True
                i += 1
            end = i
            if is_example:
                print(tokens[idx])
                print(dep_indices[idx])
                print(new_ypred[start:end])
                print(new_ytrue[start:end])
    return rows, pos_drop, neg_drop


class TreeConstructor:
    def __init__(self):
        pass

    @staticmethod
    def validate_tree(tree):
        sources, targets = tree
        if len(sources) != len(targets):
            raise ValueError("Sources and targets must have equal length")
        return True

    @staticmethod
    def validate_tree_list(trees):
        for i, tree in enumerate(trees):
            try:
                TreeConstructor.validate_tree(tree)
            except ValueError as e:
                raise ValueError(f"Invalid tree at index {i}: {e}")

    @staticmethod
    def get_edges(tree):
        sources, targets = tree
        return list(zip(sources, targets))

    @staticmethod
    def get_all_edges(trees):
        return [TreeConstructor.get_edges(tree) for tree in trees]

    @staticmethod
    def calculate_node_depths(tree):
        targets, sources = tree

        # Build adjacency list for the graph
        graph = defaultdict(list)
        for src, tgt in zip(sources, targets):
            graph[src].append(tgt)

        # Find nodes with incoming edges
        incoming = set(targets)

        # Identify root nodes (nodes with no incoming edges)
        all_nodes = set(sources) | set(targets)
        roots = [node for node in all_nodes if node not in incoming]

        # Initialize depths
        depths = {node: 0 for node in all_nodes}

        # Function to compute depth via DFS
        def compute_depth(node, current_depth):
            # Update depth if this path is longer
            depths[node] = max(depths[node], current_depth)
            # Explore children
            for neighbor in graph[node]:
                compute_depth(neighbor, current_depth + 1)

        # Compute depths starting from each root
        for root in roots:
            compute_depth(root, 0)

        return depths

    @staticmethod
    def calculate_depths_for_trees(trees):
        return [TreeConstructor.calculate_node_depths(tree) for tree in trees]