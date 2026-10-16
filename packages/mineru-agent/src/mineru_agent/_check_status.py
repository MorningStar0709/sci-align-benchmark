import json
cp = json.load(open(r"benchmark/agent/results/visual_audit_checkpoint.json", encoding="utf-8"))
s = cp["summary"]
print(f"进度: {cp['processed']}/{cp['total']}")
print(f"一致:     {s['consistent']} ({s['consistent_pct']}%)")
print(f"部分一致: {s['partially_consistent']} ({s['partially_consistent_pct']}%)")
print(f"不一致:   {s['inconsistent']} ({s['inconsistent_pct']}%)")
print(f"不确定:   {s['uncertain']} ({s['uncertain_pct']}%)")
print(f"错误:     {s['error_count']}")
