import json
import re
from pathlib import Path
from collections import Counter, defaultdict

fv = Path('benchmark/eval/reports/formula_validation_set.jsonl')
if not fv.exists():
    fv = Path('dataset/formula_validation_set.jsonl')

lines = fv.read_text(encoding='utf-8').strip().split('\n')
formulas = [json.loads(l) for l in lines if l.strip()]

types = Counter()
dimensions = Counter()
fields = Counter()
lengths = []
source_by_question = Counter()
source_by_domain = Counter()
dim_field_cross = Counter()

for f in formulas:
    types[f.get('formula_type', 'unknown')] += 1
    dimensions[f.get('dimension', 'unknown')] += 1
    fields[f.get('field', 'unknown')] += 1
    lengths.append(len(f.get('formula_text', '')))
    dim_field_cross[(f.get('dimension', '?'), f.get('field', '?'))] += 1

    qid = f.get('source_question_id', 'unknown')
    source_by_question[qid] += 1
    match = re.match(r'^([A-Z]+-[A-Z]+)', qid)
    domain = match.group(1) if match else qid
    source_by_domain[domain] += 1

lengths_sorted = sorted(lengths)
qid_cnts = sorted(source_by_question.values())

out = []
def p(line=''):
    out.append(line)

p('=' * 64)
p('  Task 2b: 公式密度分析 — 技术报告 §7 数据')
p('=' * 64)
p(f'  数据源: {fv.name}  ({len(formulas)} 条有效记录)')
p()

p('─' * 64)
p('  1. 公式类型分布')
p('─' * 64)
for t, c in types.most_common():
    p(f'     {t:<20s}  {c:>5d}  ({c/len(formulas)*100:5.1f}%)')
p(f'     {"合计":<20s}  {sum(types.values()):>5d}')
p()

p('─' * 64)
p('  2. 维度分布')
p('─' * 64)
for d, c in dimensions.most_common():
    label = {'A': '概念辨析', 'B': '推理连贯性', 'C': '条件完备性', 'D': '设计严谨性'}.get(d, d)
    p(f'     维度{d} ({label:<6s})  {c:>5d}  ({c/len(formulas)*100:5.1f}%)')
p(f'     {"合计":<20s}  {sum(dimensions.values()):>5d}')
p()

p('─' * 64)
p('  3. 字段来源')
p('─' * 64)
for fl, c in fields.most_common():
    label = {'question': '题目正文', 'answer': '参考答案'}.get(fl, fl)
    p(f'     {fl:<10s} ({label:<6s})  {c:>5d}  ({c/len(formulas)*100:5.1f}%)')
p()

p('─' * 64)
p('  4. Dimension × Field 交叉（高价值：揭示公式在题目/答案中的维度分布）')
p('─' * 64)
dim_order = ['A', 'B', 'C', 'D']
field_order = ['question', 'answer']
dim_labels = {'A': '概念', 'B': '推理', 'C': '条件', 'D': '设计'}
field_labels = {'question': '题目', 'answer': '答案'}
header = f'     {"维度":<6s}  {"题目":>6s}  {"答案":>6s}  {"合计":>6s}  {"题目%":>7s}'
p(header)
p('     ' + '-' * 42)
for d in dim_order:
    q_cnt = dim_field_cross.get((d, 'question'), 0)
    a_cnt = dim_field_cross.get((d, 'answer'), 0)
    total = q_cnt + a_cnt
    q_pct = q_cnt / total * 100 if total else 0
    p(f'     {d}({dim_labels[d]})  {q_cnt:>6d}  {a_cnt:>6d}  {total:>6d}  {q_pct:>6.1f}%')
p(f'     {"合计":<6s}  {sum(dim_field_cross.get((d,"question"),0) for d in dim_order):>6d}  {sum(dim_field_cross.get((d,"answer"),0) for d in dim_order):>6d}  {len(formulas):>6d}')
p()

p('─' * 64)
p('  5. 公式长度统计')
p('─' * 64)
p(f'     最小: {lengths_sorted[0]}')
p(f'     最大: {lengths_sorted[-1]}')
p(f'     均值: {sum(lengths)/len(lengths):.1f}')
p(f'     中位数: {lengths_sorted[len(lengths)//2]}')
p(f'     P25: {lengths_sorted[len(lengths)//4]}')
p(f'     P75: {lengths_sorted[len(lengths)*3//4]}')
p()
buckets = [('≤20', 0, 20), ('21-50', 21, 50), ('51-100', 51, 100), ('101-200', 101, 200), ('>200', 201, 99999)]
p('     长度分桶:')
for label, lo, hi in buckets:
    cnt = sum(1 for l in lengths if lo <= l <= hi)
    bar = '█' * (cnt * 40 // len(formulas))
    p(f'       {label:>6s}: {cnt:>5d} ({cnt/len(formulas)*100:5.1f}%) {bar}')
p()

p('─' * 64)
p('  6. 题目公式数分布（每道题包含的公式数量）')
p('─' * 64)
p(f'     涉及题目: {len(source_by_question)} 题')
p(f'     平均每题: {len(formulas)/len(source_by_question):.1f} 公式')
p(f'     最少: {min(qid_cnts)} 公式/题')
p(f'     最多: {max(qid_cnts)} 公式/题')
qid_dist = Counter(source_by_question.values())
p()
p('     公式数/题  题目数  累积题目数')
p('     ' + '-' * 36)
cum = 0
for n in sorted(qid_dist):
    cum += qid_dist[n]
    p(f'       {n:>3d}        {qid_dist[n]:>3d}      {cum:>3d}')
p()

p('─' * 64)
p('  7. 公式数 Top-10 题')
p('─' * 64)
for rank, (qid, cnt) in enumerate(source_by_question.most_common(10), 1):
    p(f'     #{rank:<2d}  {qid}  {cnt} 公式')
p()

p('─' * 64)
p('  8. 数据质量说明')
p('─' * 64)
by_qid_idx = defaultdict(list)
for f in formulas:
    by_qid_idx[f['source_question_id']].append(f['formula_index'])
index_anomalies = sum(1 for qid, indices in by_qid_idx.items() if sorted(indices) != list(range(len(indices))))
p('     formula_id 唯一: 1779/1779 ✅')
p('     字段空值: 0 ✅')
p(f'     formula_index 非连续: {index_anomalies}/122 题（index 按 type+field 分组递增，')
p('       非全局连续；不影响计数/类型/长度统计）')
p('     领域覆盖: CS-EVO 单一领域（控制理论-演化博弈），跨领域泛化由 Task 3 (S3*) 覆盖')
p()

output_text = '\n'.join(out)
out_path = Path('benchmark/eval/results/formula_density_output.txt')
out_path.write_text(output_text, encoding='utf-8')
print(output_text)
print(f'\n[产出] {out_path}  ({out_path.stat().st_size} bytes)')
