"""manifest.yaml 一致性检查（T3.1 验证，临时脚本）。

检查项：
    1. variants 的 model_file_ids 与 model_files 声明完全一致（无悬空引用）
    2. default_variant 在 variants 列表中
    3. required_for 双向一致（file.required_for ⊆ 引用该文件的 variants）
    4. exclusive_group 成员数（两组各 3 档）
    5. 每个 variant 恰好含 t2i + kontext 各一档 transformer
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.yaml"


def main() -> int:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    models = data["models"]
    files = {f["id"]: f for f in models["model_files"]}
    ids = set(files.keys())
    print(f"engine_id: {data['engine_id']}")
    print(f"paradigm: {data['paradigm']}")
    print(f"model_files 总数: {len(files)}")
    print(f"default_variant: {models['default_variant']}")

    ok = True
    variant_names = set()
    for v in models["variants"]:
        variant_names.add(v["name"])
        vf = set(v["model_file_ids"])
        missing = vf - ids
        if missing:
            print(f"  [FAIL] variant {v['name']} 引用不存在的 file_id: {sorted(missing)}")
            ok = False

    if models["default_variant"] not in variant_names:
        print("[FAIL] default_variant 不在 variants 列表")
        ok = False

    # required_for 双向一致
    for fid, f in files.items():
        for rv in f.get("required_for") or []:
            if rv not in variant_names:
                print(f"[FAIL] {fid} required_for 引用不存在 variant: {rv}")
                ok = False
            else:
                v = next(x for x in models["variants"] if x["name"] == rv)
                if fid not in v["model_file_ids"]:
                    print(f"[FAIL] {fid} required_for 声明 {rv}，但该 variant 未引用它")
                    ok = False

    # exclusive_group 核对
    groups = defaultdict(list)
    for fid, f in files.items():
        g = f.get("exclusive_group")
        if g:
            groups[g].append(fid)
    for g, members in sorted(groups.items()):
        print(f"  exclusive_group {g}: {len(members)} 个成员 -> {sorted(members)}")

    # 孤儿文件（无任何 variant 引用）
    total_refs = set()
    for v in models["variants"]:
        total_refs |= set(v["model_file_ids"])
    orphan = ids - total_refs
    if orphan:
        print(f"[WARN] 无任何 variant 引用的文件: {sorted(orphan)}")

    # 每个 variant 必须含两组 transformer 各一档
    for v in models["variants"]:
        t2i = [i for i in v["model_file_ids"] if i.startswith("flux-t2i-transformer-")]
        kt = [i for i in v["model_file_ids"] if i.startswith("flux-kontext-transformer-")]
        shared = len(v["model_file_ids"]) - len(t2i) - len(kt)
        print(f"  {v['name']}: t2i={len(t2i)} kontext={len(kt)} shared={shared} 精度={v['precision']}")
        if len(t2i) != 1 or len(kt) != 1:
            print(f"  [FAIL] {v['name']} transformer 组合异常")
            ok = False

    # target_dir 必须在 image/ 下（model-placement-rules.md 规则）
    for fid, f in files.items():
        if not f.get("target_dir", "").startswith("image/"):
            print(f"[FAIL] {fid} target_dir 不符合 image/ 顶层规则: {f.get('target_dir')}")
            ok = False

    print()
    print("=== YAML 一致性检查:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
