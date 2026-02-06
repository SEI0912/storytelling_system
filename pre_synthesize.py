# pre_synthesize.py: 全ページの平均を算出した後に音声合成を行う
import json
import os
import textwrap
import time
from robottools3 import RobotTools 

# ==============================================================================
# 1. 設定
# ==============================================================================

DUMMY_IP = '0.0.0.0'
DUMMY_PORT = 0 
rt = RobotTools(DUMMY_IP, DUMMY_PORT) 

# ★★★ 読み聞かせたい絵本IDを設定 ★★★
CURRENT_BOOK_ID = "inu" 
STORY_FILE_PATH = f'story_{CURRENT_BOOK_ID}_normal.json' 

# キャッシュディレクトリ名
BOOK_CACHE_DIR = f'{CURRENT_BOOK_ID}_1_2_speech_cache' 
os.makedirs(BOOK_CACHE_DIR, exist_ok=True) 

# ==============================================================================
# 2. メイン処理
# ==============================================================================

# JSONファイルの読み込み
story_data = []
try:
    with open(STORY_FILE_PATH, 'r', encoding='utf-8') as f:
        story_data = json.load(f)
    print(f"✅ '{STORY_FILE_PATH}'からデータを読み込みました。全 {len(story_data)} 項目。")
except FileNotFoundError:
    print(f"❌ エラー: '{STORY_FILE_PATH}' が見つかりません。")
    exit()

# ------------------------------------------------------------------------------
# STEP A: 【重要】全ページのめくり時間を先にスキャンして平均を出す
# ------------------------------------------------------------------------------
# 実際に読み上げる「奇数ページ」のみを対象に、flip_durationをリスト化
all_flip_durations = [
    item.get('flip_duration', 600) 
    for item in story_data 
    if str(item.get('page_number', '')).isdigit() and int(item.get('page_number')) % 2 != 0
]

if not all_flip_durations:
    print("❌ 読み上げ対象のページ（奇数ページ）が見つかりませんでした。")
    exit()

# 全体平均を算出
avg_flip = sum(all_flip_durations) / len(all_flip_durations)

print("-" * 30)
print(f"📊 統計情報:")
print(f"  対象ページ数: {len(all_flip_durations)}")
print(f"  全体の平均めくり時間: {avg_flip:.1f} ms")
print("-" * 30)

# ------------------------------------------------------------------------------
# STEP B: 算出した全体平均を基に、各ページのスピードを計算して音声合成
# ------------------------------------------------------------------------------
for i, item in enumerate(story_data):
    
    text = item.get('text', '')
    page_number_raw = item.get('page_number', '')
    valence = item.get('valence', 0.0)
    intensity = item.get('intensity', 0.0)
    flip_dur = item.get('flip_duration', 600)

    # ページ番号のチェック
    if not str(page_number_raw).isdigit():
        continue
    page_number = int(page_number_raw)

    # 奇数ページのみ処理
    if page_number % 2 == 0:
        continue

    # --- 話速 (Speed) の動的計算 ---
    # 全体平均(avg_flip)とこのページのめくり時間(flip_dur)の比率
    ratio = avg_flip / flip_dur
    
    # 標準100を基準に調整。係数50で変化の幅を調整（100 + (比率-1)*50）
    # 例: 平均の2倍の長さ(比率0.5)なら Speed 75 / 平均の半分(比率2.0)なら Speed 150
    calculated_speed = 100 + (ratio - 1.0) * 40
    
    # 聴き取りやすさの限界値 (80〜140) に収める
    final_speed = max(50, min(100, int(calculated_speed)))

    formatted_page_num = str(page_number).zfill(2)
    base_filename = f"page{formatted_page_num}"
    
    print(f"[{i+1}/{len(story_data)}] Page {page_number}:")
    print(f"  めくり時間: {flip_dur}ms (全体平均との比: {ratio:.2f})")
    print(f"  → 決定されたSpeed: {final_speed}")
    
    # 音声合成の実行（rt.synthesize_and_cache_text が speed 引数を受け取る前提）
    rt.synthesize_and_cache_text(
        text=text, 
        valence=valence, 
        intensity=intensity, 
        cache_dir=BOOK_CACHE_DIR, 
        book_id=CURRENT_BOOK_ID,
        base_filename=base_filename,
        speed=final_speed 
    )

print("-" * 30)
print(f"✅ 全プロセスの完了。全体平均 {avg_flip:.1f}ms に基づき、全音声の生成が終わりました。")