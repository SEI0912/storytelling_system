import json
import random

# ===============================
# 1. ランダム 20 個生成
# ===============================
candidates = list(range(600, 4501, 100))  # 600〜4500, 100刻み → 40個
random_values = random.sample(candidates, 20)  # 重複なしで20個選ぶ

print("生成されたランダム flip_duration:")
print(random_values)
print("-" * 40)

# ===============================
# 2. JSON を読み込む
# ===============================
INPUT_JSON = "story_suhu_random.json"     # ← 添付ファイルと同名にしてください
OUTPUT_JSON = "story_suhu_random_out.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# ===============================
# 3. flip_duration を20個の乱数で上書き
# ===============================
if len(data) != 20:
    print(f"警告: JSON の項目数が20ではありません（{len(data)}項目）")

for i, item in enumerate(data):
    item["flip_duration"] = random_values[i]

# ===============================
# 4. 新しい JSON として保存
# ===============================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ 書き換え完了！ → {OUTPUT_JSON} に保存しました")
