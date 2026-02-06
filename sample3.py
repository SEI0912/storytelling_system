# sample3.py: Sotaで絵本を読み聞かせるメインの実行ファイル
import json
import os
import textwrap
import time
import re
import requests # HTTPリクエスト用にrequestsをインポート
from robottools3 import RobotTools 

# ==============================================================================
# 1. 設定
# ==============================================================================

# SotaのIPアドレスとポート番号を設定
# Sotaに接続しない場合は、このままでも音声再生部分以外は実行可能
SOTA_IP = '192.168.1.147' # Sotaの実際のIPアドレスに置き換えてください
SOTA_PORT = 22222 

rt = RobotTools(SOTA_IP, SOTA_PORT, audio_port=30001, use_audio_ack=True) 

import robottools3
print("✅ using robottools3 from:", robottools3.__file__)
print("✅ use_audio_ack =", getattr(rt, "_RobotTools__use_audio_ack", "UNKNOWN"))
print("✅ audio_port =", getattr(rt, "_RobotTools__audio_port", "UNKNOWN"))
print("✅ ip =", getattr(rt, "_RobotTools__ip", "UNKNOWN"))


# ★★★ 読み聞かせたい絵本IDを設定 ★★★
CURRENT_BOOK_ID = "suhu" #絵本の種類変更
STORY_FILE_PATH = f'story_{CURRENT_BOOK_ID}_normal.json' #ページめくりとロボットモーションを変更できる

# キャッシュディレクトリの設定 (pre_synthesize.pyと一致させる)
BOOK_CACHE_DIR = f'{CURRENT_BOOK_ID}_1_speech_cache' #音声を変更できる

# Flaskサーバーへの通知URL (app3.pyと同期)
FLASK_NOTIFICATION_URL = 'http://127.0.0.1:5000/sota_reading_finished' 
FLASK_FINISH_URL = 'http://127.0.0.1:5000/reading_finished'


# ==============================================================================
# 2. ヘルパー関数: 感情パラメータをポーズに変換
# ==============================================================================
def map_emotion_to_pose(valence: float, intensity: float) -> str:
    """
    感情パラメータ(V, I)に基づいて、Sotaのポーズを決定する。
    
    注: この関数は現在、ポーズ実行ロジックからは使われていませんが、
    感情の分類ロジックとして残しています。
    """
    if valence > 0.3 and intensity > 0.5:
        return "happy"  
    elif valence < -0.3 and intensity > 0.5:
        return "sad"    
    elif intensity > 0.3:
        return "interest" 
    else:
        return "neutral" 


# ==============================================================================
# 3. メイン処理：ストーリーデータのロードと読み聞かせ実行
# ==============================================================================

# JSONファイルからストーリーデータを読み込む
story_data = []
try:
    with open(STORY_FILE_PATH, 'r', encoding='utf-8') as f:
        story_data = json.load(f)
    print(f"✅ '{STORY_FILE_PATH}'からストーリーデータを読み込みました。全 {len(story_data)} ページ。")
except FileNotFoundError:
    print(f"❌ エラー: '{STORY_FILE_PATH}' が見つかりません。ファイル名を確認してください。")
    exit()
except json.JSONDecodeError:
    print(f"❌ エラー: '{STORY_FILE_PATH}' のJSON形式が不正です。")
    exit()

print("Sotaによる絵本の読み聞かせを開始します。Enterを押してください。")
input() 


print("-" * 30)
print("📚 Sotaによる絵本読み聞かせを開始します...")
print(f"キャッシュフォルダ: {BOOK_CACHE_DIR}")
print("-" * 30)


# ==============================================================================
# 先読み（Sota側へ保存）ユーティリティ
# ==============================================================================

import threading

def _to_int_page(x, default_val: int) -> int:
    try:
        return int(x)
    except Exception:
        return default_val

def find_next_odd_page_index(story_data, start_i: int) -> int:
    """start_i より後ろで最初に見つかる奇数ページの index を返す。なければ -1。"""
    for j in range(start_i + 1, len(story_data)):
        pn = _to_int_page(story_data[j].get("page_number", j + 1), j + 1)
        if pn % 2 == 1:
            return j
    return -1

# どのページがSotaに先読み済みか（key_prefixで管理）
preloaded_prefix = set()

time.sleep(10)


# 全てのページをループして読み聞かせを実行
for i, item in enumerate(story_data):
    
    text = item.get('text', '')
    page_number_raw = item.get('page_number', i + 1)
    valence = item.get('valence', 0.0)
    intensity = item.get('intensity', 0.0)

    # ページ番号を整数に変換
    try:
        page_number = int(page_number_raw)
    except ValueError:
        print(f"⚠️ ページ番号 '{page_number_raw}' は不正な値です。スキップします。")
        continue

    # 偶数ページはスキップ（JSONのデータ構造に依存）
    if page_number % 2 == 0:
        print(f"--- ページ {page_number} は偶数ページのためスキップしました。")
        continue

    formatted_page_num = str(page_number).zfill(2)
    base_filename = f"page{formatted_page_num}"
    
    # ページ情報を表示
    print(f"\n=======================================================")
    print(f"📖 Page {page_number} の読み聞かせを開始 ({i+1}/{len(story_data)})")
    print(f"  感情: V={valence:.4f}, I={intensity:.4f}")
    print(f"  テキスト: {text.strip()}")
    print(f"  ファイルコア: {CURRENT_BOOK_ID}_{base_filename}")
    print(f"=======================================================")

    # 感情に基づくポーズの実行ロジックは削除されました。

    # 事前合成された音声ファイルを再生
    # この関数内でチャンクの連続再生と待機が行われます。
    # --- 先読み: 現ページが未送信なら、先にSotaへ保存しておく ---
    key_prefix = f"{CURRENT_BOOK_ID}_{base_filename}"
    if key_prefix not in preloaded_prefix:
        rt.preload_cached_speech_to_sota(
            base_filename=base_filename,
            cache_dir=BOOK_CACHE_DIR,
            book_id=CURRENT_BOOK_ID,
            key_prefix=key_prefix,
        )
        preloaded_prefix.add(key_prefix)

    # --- 先読み: 次ページ（奇数ページ）の音声を、現ページ再生中にSotaへ保存しておく ---
    next_idx = find_next_odd_page_index(story_data, i)
    preload_thread = None
    if next_idx != -1:
        next_item = story_data[next_idx]
        next_pn = _to_int_page(next_item.get("page_number", next_idx + 1), next_idx + 1)
        next_base = f"page{str(next_pn).zfill(2)}"
        next_prefix = f"{CURRENT_BOOK_ID}_{next_base}"
        if next_prefix not in preloaded_prefix:
            preload_thread = threading.Thread(
                target=lambda: (
                    rt.preload_cached_speech_to_sota(
                        base_filename=next_base,
                        cache_dir=BOOK_CACHE_DIR,
                        book_id=CURRENT_BOOK_ID,
                        key_prefix=next_prefix,
                    ),
                    preloaded_prefix.add(next_prefix)
                ),
                daemon=True
            )
            preload_thread.start()

    # --- 再生: 先読み済みのキーから再生（送信なし） ---
    duration = rt.play_cached_speech_from_sota(
        key_prefix=key_prefix,
        cache_dir=BOOK_CACHE_DIR,
        base_filename=base_filename,
        book_id=CURRENT_BOOK_ID
    )

    # 念のため：次ページの先読みスレッドがまだなら、この時点で継続してOK（ここではjoinしない）

    # ページ間に短いウェイトを設ける (この待機後、ページめくり通知を送信)
    if duration > 0:
        # time.sleep(1.0) # ★★★ 削除：この待機をなくすことで、めくり処理の開始を早めます ★★★
        pass 
    
    # ページめくり通知の判定
    is_last_item = (i == len(story_data) - 1)
    
    # ページめくりが必要な条件: 奇数ページであり、最後の項目ではない
    should_flip = (page_number % 2 != 0) and (not is_last_item)

    if should_flip:
        
        # 今読み終わったページ (item) のめくり時間を使用する
        current_page_flip_duration = item.get('flip_duration', 600)
        
        # ★★★ 修正箇所: 動的モーション時間計算ロジックを再挿入 ★★★
        
        # 2. モーション片道時間 (T_motion_half) を計算
        T_lag = 0 # 想定されるラグ時間 (ms)
        # 計算結果が100ms未満にならないよう、max(100, ...)で下限を設けます
        T_motion_half = max(100, (current_page_flip_duration - T_lag) // 2)
        
        print(f'  めくり時間 T_flip: {current_page_flip_duration}ms')
        print(f'  モーション片道時間 T_motion_half: {T_motion_half}ms を採用します。')
        
        # 3. モーションのMsecを動的に計算されたT_motion_halfに置き換え、実行
        nod_motion = [
             dict(Msec=T_motion_half, ServoMap=dict(BODY_Y=60, L_SHOU=-90, L_ELBO=0, R_SHOU=30, R_ELBO=20, HEAD_Y=30, HEAD_P=0, HEAD_R=0 )),
             dict(Msec=T_motion_half, ServoMap=dict(BODY_Y=0, L_SHOU=-90, L_ELBO=0, R_SHOU=90, R_ELBO=0, HEAD_Y=0, HEAD_P=0, HEAD_R=0 ))
         ] 
        print('🤖 ページめくりモーション (動的Msec) を実行します。')
        rt.play_motion(nod_motion) # play_motionが完了まで待機することを前提とします。
        
        # モーション実行ロジックは削除されています。

        print(f'📢 Flaskサーバーにページめくり通知を送信します。めくり速度: {current_page_flip_duration}ms')
        
        try:
            # Flaskサーバーへのめくり通知 (POSTリクエスト)
            response = requests.post(FLASK_NOTIFICATION_URL, json={'flip_duration': current_page_flip_duration}) 
            if response.status_code == 200:
                print('✅ Flaskサーバーへの通知に成功しました。')
            else:
                 print(f'⚠️ Flaskサーバーへの通知に失敗しました。ステータスコード: {response.status_code}')
        except requests.exceptions.ConnectionError as e:
            # Flaskサーバーが起動していない、またはURLが間違っている可能性
            print(f'❌ Flaskサーバーへの接続に失敗しました。サーバーが起動しているか、URL ({FLASK_NOTIFICATION_URL}) が正しいか確認してください: {e}')
        except Exception as e:
            print(f'❌ ページめくり通知中に予期せぬエラー: {e}')
        
        # 5. ページめくりコマンド送信後に待機 (ブラウザめくり完了を待つ)
        # ★★★ 修正箇所: めくり時間（ms）を秒に変換して待機 ★★★
        sleep_duration_sec = current_page_flip_duration / 1000.0
        print(f'⏳ ブラウザのめくり完了を待機します: {sleep_duration_sec:.2f}秒')
        time.sleep(sleep_duration_sec)
            
    # ページめくり通知の判定ここまで

print("-" * 30)
print("🔚 読み聞かせを終了しました。")
print("-" * 30)

# ===== 読み聞かせ終了後：アンケート表示をFlaskへ通知 =====
SURVEY_URL_1 = "https://docs.google.com/forms/d/e/1FAIpQLSfDX3h59_ZCFEYGNTfqPsHqRfY69Vvbhs5AvI-PHrpjmsIesA/viewform?usp=header"  # ←ここを自分のフォームURLに
SURVEY_URL_2 = "https://docs.google.com/forms/d/e/1FAIpQLScJa9IvHXWeEa_lO8a_kEe0IlFt0nVLH93FTqgIKGI0opZtug/viewform?usp=header"
SURVEY_URL_3 = "https://docs.google.com/forms/d/e/1FAIpQLSfgqHEOcm5HBwXZKY2FUmc_kDqEg7NxzO1mVO3hLmmoy13fcg/viewform?usp=header"

if BOOK_CACHE_DIR == f'{CURRENT_BOOK_ID}_1_speech_cache':
    SURVEY_URL = SURVEY_URL_1
elif BOOK_CACHE_DIR == f'{CURRENT_BOOK_ID}_2_speech_cache':
    SURVEY_URL = SURVEY_URL_2
elif BOOK_CACHE_DIR == f'{CURRENT_BOOK_ID}_3_speech_cache':
    SURVEY_URL = SURVEY_URL_3


try:
    r = requests.post(FLASK_FINISH_URL, json={"survey_url": SURVEY_URL}, timeout=3)
    print(f"✅ アンケート表示通知を送信しました: {r.status_code}")
except Exception as e:
    print(f"❌ アンケート表示通知に失敗: {e}")
