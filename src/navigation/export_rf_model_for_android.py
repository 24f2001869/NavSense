"""
SIH26168 - Export Random Forest Model to Standalone Java/Kotlin Evaluator
Script: src/navigation/export_rf_model_for_android.py

Trains the causal Random Forest speed model (35 trees, depth 8) and exports
both JSON metadata and a zero-dependency CausalSpeedModel.java class for Android.
"""

import sys
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_phone_speed_c8_7 import prepare_trip_phone_data

MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
JAVA_ENGINE_DIR = REPO_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "sih26168" / "idr" / "engine"
JAVA_ENGINE_DIR.mkdir(parents=True, exist_ok=True)


def export_rf():
    print("1. Ingesting Vta02 and training Random Forest speed model...")
    d_vta02 = prepare_trip_phone_data('Vta02')
    rf = RandomForestRegressor(n_estimators=35, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(d_vta02['features'], d_vta02['gt_speed'])

    # Export to JSON
    trees_json = []
    for est in rf.estimators_:
        t = est.tree_
        trees_json.append({
            'feature': t.feature.tolist(),
            'threshold': [round(float(x), 6) for x in t.threshold],
            'children_left': t.children_left.tolist(),
            'children_right': t.children_right.tolist(),
            'value': [round(float(x[0, 0]), 6) for x in t.value]
        })

    json_path = MODELS_DIR / "rf_speed_model.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({'n_estimators': len(trees_json), 'n_features': 16, 'trees': trees_json}, f)
    print(f"Saved JSON tree manifest: {json_path}")

    # Generate standalone CausalSpeedModel.java
    print("2. Generating zero-dependency CausalSpeedModel.java for Android...")
    java_code = [
        "package com.sih26168.idr.engine;",
        "",
        "/**",
        " * Auto-generated embedded Random Forest speed predictor.",
        " * 35 decision trees, max depth 8, trained on Ford Fiesta highway trip.",
        " * Zero external dependencies, pure Java/Kotlin execution (< 0.05 ms per prediction).",
        " */",
        "public class CausalSpeedModel {",
        "",
        "    public static class DecisionTree {",
        "        public final int[] feature;",
        "        public final double[] threshold;",
        "        public final int[] childrenLeft;",
        "        public final int[] childrenRight;",
        "        public final double[] value;",
        "",
        "        public DecisionTree(int[] feature, double[] threshold, int[] childrenLeft, int[] childrenRight, double[] value) {",
        "            this.feature = feature;",
        "            this.threshold = threshold;",
        "            this.childrenLeft = childrenLeft;",
        "            this.childrenRight = childrenRight;",
        "            this.value = value;",
        "        }",
        "",
        "        public double predict(double[] x) {",
        "            int node = 0;",
        "            while (childrenLeft[node] != -1) {",
        "                int f = feature[node];",
        "                if (x[f] <= threshold[node]) {",
        "                    node = childrenLeft[node];",
        "                } else {",
        "                    node = childrenRight[node];",
        "                }",
        "            }",
        "            return value[node];",
        "        }",
        "    }",
        "",
        "    private final DecisionTree[] trees;",
        "",
    ]

    chunk_size = 7
    n_chunks = (len(trees_json) + chunk_size - 1) // chunk_size

    java_code.append("    public CausalSpeedModel() {")
    java_code.append(f"        this.trees = new DecisionTree[{len(trees_json)}];")
    for g in range(n_chunks):
        java_code.append(f"        initGroup{g}();")
    java_code.append("    }")
    java_code.append("")

    for g in range(n_chunks):
        java_code.append(f"    private void initGroup{g}() {{")
        start_idx = g * chunk_size
        end_idx = min(len(trees_json), (g + 1) * chunk_size)
        for idx in range(start_idx, end_idx):
            t = trees_json[idx]
            feat_str = "new int[]{" + ",".join(map(str, t['feature'])) + "}"
            thresh_str = "new double[]{" + ",".join(map(str, t['threshold'])) + "}"
            left_str = "new int[]{" + ",".join(map(str, t['children_left'])) + "}"
            right_str = "new int[]{" + ",".join(map(str, t['children_right'])) + "}"
            val_str = "new double[]{" + ",".join(map(str, t['value'])) + "}"
            java_code.append(f"        trees[{idx}] = new DecisionTree({feat_str}, {thresh_str}, {left_str}, {right_str}, {val_str});")
        java_code.append("    }")
        java_code.append("")

    java_code.extend([
        "    public double predict(double[] features) {",
        "        if (features == null || features.length < 16) return 0.0;",
        "        double sum = 0.0;",
        "        for (DecisionTree tree : trees) {",
        "            sum += tree.predict(features);",
        "        }",
        "        double pred = sum / trees.length;",
        "        return Math.max(0.0, pred);",
        "    }",
        "}",
        ""
    ])

    java_path = JAVA_ENGINE_DIR / "CausalSpeedModel.java"
    with open(java_path, "w", encoding="utf-8") as f:
        f.write("\n".join(java_code))
    print(f"Generated standalone Android class: {java_path}")


if __name__ == "__main__":
    export_rf()
