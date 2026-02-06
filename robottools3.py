# robottools3.py (修正版: チャンク間ギャップ最小化)
import json
import socket
import typing
import os
import wave
import time 
import random
import textwrap 
import re 
import hashlib 
from voicepeak_cli_min import synth 

# ★ 修正: SPEECH_CACHE_DIR のグローバル定義は削除し、各関数で動的に扱う ★

class RobotTools(object):
    def __init__(self, ip: str, port: int, audio_port: int = 30001, use_audio_ack: bool = False):
        """
        ip/port: 既存のRobotToolsサーバ（通常 22222）への接続先。
        audio_port: Sota上で動かすACK付き音声サーバのポート（例: 30001）。
        use_audio_ack: True の場合、音声再生は audio_port 側へ送り、ACKを受け取るまで待機する。
        """
        self.__ip = ip
        self.__port = port
        self.__audio_port = audio_port
        self.__use_audio_ack = use_audio_ack

    def set_audio_ack_enabled(self, enabled: bool) -> None:
        """ACK付き音声再生を有効/無効にする。"""
        self.__use_audio_ack = bool(enabled)

    # --- ヘルパー: キャッシュ用ファイル名生成 (cache_dir, book_id を追加) ---
    def _get_cache_path(self, text: str, valence: float, intensity: float, narrator: str, 
                        chunk_index: int, total_chunks: int, cache_dir: str, 
                        base_name: typing.Optional[str] = None, book_id: typing.Optional[str] = None,
                        speed=100) -> str:
        """
        ファイルパスを生成する。cache_dirを使用し、ファイル名にbook_idをプレフィックスとして付与する。
        """
        # ハッシュ元の文字列を生成
        unique_string = f"{text}|{valence}|{intensity}|{narrator}|{speed}"
        # SHA256でハッシュ値を計算 (8文字に短縮)
        hash_id = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:8] 
        
        # ファイル名のコアに book_id を追加
        file_prefix = f"{book_id}_" if book_id else ""
        
        if base_name:
            sanitized_name = base_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            
            if total_chunks > 1:
                # 複数チャンクの場合: suhu_page03_1__hash.wav
                filename_core = f"{file_prefix}{sanitized_name}_{chunk_index + 1}" 
            else:
                # 単一チャンクの場合: suhu_page01__hash.wav
                filename_core = f"{file_prefix}{sanitized_name}"
                
            # 命名規則に基づく名前とハッシュを結合
            wav_filename = f"{filename_core}__{hash_id}.wav" 
        else:
            # base_nameがない場合 (say_text_with_emotion用の一時キャッシュ)
            wav_filename = f"{file_prefix}{hash_id}.wav" 

        # 渡された cache_dir を使用
        return os.path.join(cache_dir, wav_filename)

    # --- ヘルパー: キャッシュファイルの検索とソート (cache_dir, book_id を追加) ---
    def _get_cached_chunk_files(self, base_prefix: str, cache_dir: str, book_id: typing.Optional[str] = None) -> typing.List[str]:
        """
        指定されたベースファイル名に一致するキャッシュファイルを検索し、ソートしてフルパスのリストを返す。
        """
        target_files = []
        # 指定された cache_dir をリスト化
        all_files = os.listdir(cache_dir)
        
        # 検索パターンに book_id プレフィックスを考慮
        safe_prefix = re.escape(base_prefix)
        file_prefix = re.escape(f"{book_id}_") if book_id else ""
        
        # ハッシュ値の部分 (\w{8}) の前に、速度情報が含まれる可能性があるため、
        # どのような文字列が来ても一致するように .* を入れます
        pattern = re.compile(rf"{file_prefix}({safe_prefix}|{safe_prefix}_\d+)__.*\.wav$")

        for filename in all_files:
            if pattern.match(filename):
                target_files.append(os.path.join(cache_dir, filename))
        
        # ソートロジック
        def sort_key(filepath):
            filename = os.path.basename(filepath)
            # book_id プレフィックスとハッシュ値を削除
            name_part = filename.split('__')[0]
            if book_id and name_part.startswith(f"{book_id}_"):
                 name_part = name_part[len(f"{book_id}_"):]
            
            # ページ番号 (XX) とチャンク番号 (N) を抽出
            match = re.search(r'page(\d+)(?:_(\d+))?', name_part)
            
            if match:
                page_num = int(match.group(1)) 
                chunk_num = int(match.group(2)) if match.group(2) else 0 
                return (page_num, chunk_num)
            
            return (99999, 99999) 

        target_files.sort(key=sort_key)
        return target_files

    # --- play_cached_speech (事前合成ファイルの再生専用関数) ---
    def play_cached_speech(self, base_filename: str, cache_dir: str, book_id: typing.Optional[str] = None) -> float:
        """
        事前合成キャッシュwavを再生する。
        - use_audio_ack=True のとき：
            ページ内の複数wavをまとめてSotaへ送信（BATCH）し、最後まで再生完了したらACKを受け取る。
            → ファイル間ラグがほぼ消える
        - use_audio_ack=False のとき：従来通り1つずつ送ってsleep
        """
        print(f"【事前合成モード】キャッシュディレクトリ'{cache_dir}'内のファイル再生を開始します。")
        chunk_files = self._get_cached_chunk_files(base_filename, cache_dir, book_id)
        total_duration = 0.0

        if not chunk_files:
            print(f"【エラー】'{book_id}_{base_filename}' に対応するキャッシュファイルが'{cache_dir}'に見つかりません。")
            return 0.0

        # ========== ACK(BATCH)方式 ==========
        if self.__use_audio_ack:
            items: typing.List[typing.Tuple[int, bytes]] = []  # (duration_ms, wav_bytes)
            per_file_info: typing.List[typing.Tuple[str, float, int]] = []  # (display_name, sec, ms)

            for wav_filename in chunk_files:
                if not os.path.exists(wav_filename):
                    print(f"【エラー】ファイルが見つかりません: {wav_filename}")
                    continue

                try:
                    # wavバイト列
                    with open(wav_filename, "rb") as f:
                        wav_bytes = f.read()

                    # duration算出（ms）
                    with wave.open(wav_filename, "rb") as wf:
                        fr = wf.getframerate()
                        nf = wf.getnframes()
                        sec = (nf / fr) if fr > 0 else 0.0
                    duration_ms = int(sec * 1000.0 + 0.999)  # ceil相当

                    display_name = os.path.basename(wav_filename).split("__")[0]
                    if book_id:
                        display_name = display_name.replace(f"{book_id}_", "")

                    items.append((duration_ms, wav_bytes))
                    per_file_info.append((display_name, sec, duration_ms))
                    total_duration += sec

                except Exception as e:
                    print(f"音声ファイルの読み込み中にエラーが発生しました ({os.path.basename(wav_filename)}): {e}")
                    continue

            if not items:
                print("【エラー】再生できる音声がありませんでした。")
                return 0.0

            # まとめて再生（最後まで終わったらACKが返る）
            print(f"▶ [ACK-BATCH] {len(items)} files will be played continuously.")
            # まとめて再生（最後まで終わったらACKが返る）
            if len(items) == 1:
                duration_ms, wav_bytes = items[0]
                print("▶ [ACK] single file (PLAY) mode.")
                self.play_wav_data_ack(wav_bytes, duration_ms=duration_ms)
            else:
                print(f"▶ [ACK-BATCH] {len(items)} files will be played continuously.")
                self.play_wav_batch_ack(items)


            # ACKを受け取った後にまとめてログ表示（個別完了は確認できないので注意）
            for i, (name, sec, ms) in enumerate(per_file_info, 1):
                print(f"  ✓ queued {i}/{len(per_file_info)}: {name}.wav  ({sec:.2f}s, {ms}ms)")

            print(f"すべての音声ファイルの再生が完了しました。合計再生時間: {total_duration:.2f}秒")
            time.sleep(0.05)
            return total_duration

        # ========== 従来方式（ACKなし） ==========
        for i, wav_filename in enumerate(chunk_files):
            if not os.path.exists(wav_filename):
                print(f"【エラー】ファイルが見つかりません: {wav_filename}")
                continue

            try:
                with open(wav_filename, "rb") as f:
                    full_wav_data = f.read()

                with wave.open(wav_filename, "rb") as wf:
                    frame_rate = wf.getframerate()
                    num_frames = wf.getnframes()
                    audio_duration = num_frames / frame_rate if frame_rate > 0 else 0.0

                # RobotToolsサーバへ送信（従来）
                t0 = time.monotonic()
                self.play_wav_data(full_wav_data)
                send_elapsed = time.monotonic() - t0
                safety_margin = 0.25
                time.sleep(audio_duration + send_elapsed + safety_margin)

                total_duration += audio_duration

                display_name = os.path.basename(wav_filename).split("__")[0]
                if book_id:
                    display_name = display_name.replace(f"{book_id}_", "")

                print(
                    f"チャンク {i+1}/{len(chunk_files)} の再生が完了しました。"
                    f"再生時間: {audio_duration:.2f}秒 (ファイル: {display_name}.wav)"
                )

            except Exception as e:
                print(f"音声ファイルの再生中にエラーが発生しました ({os.path.basename(wav_filename)}): {e}")
                continue

        print(f"すべての音声ファイルの再生が完了しました。合計再生時間: {total_duration:.2f}秒")
        time.sleep(0.05)
        return total_duration


    

    # --- synthesize_and_cache_text (事前音声合成用) ---
    # --- synthesize_and_cache_text (事前音声合成用) ---
    def synthesize_and_cache_text(self, text: str, valence: float = 0.0, intensity: float = 0.0, 
                                base_filename: typing.Optional[str] = None, 
                                cache_dir: typing.Optional[str] = 'speech_cache',
                                book_id: typing.Optional[str] = None,
                                speed = 100) -> None:
        """
        VOICEPEAK CLIを使用して音声を生成し、キャッシュに保存する。再生は行わない。
        """
        print(f"--- プリロード開始: テキストの音声合成を開始 ---")
        narrator = "Japanese Female 1" 
        
        # --- テキストを句読点で分割（say_text_with_emotion と同一ロジックを流用） ---
        MAX_CHARS = 140 
        segments = re.split(r'(?<=[。！？\n])', text)
        text_chunks = []
        current_chunk = ""
        for segment in segments:
            if not segment.strip():
                continue
            if len(current_chunk) + len(segment) > MAX_CHARS and current_chunk:
                text_chunks.append(current_chunk.strip())
                current_chunk = segment
            else:
                current_chunk += segment
        if current_chunk:
            if len(current_chunk) > MAX_CHARS:
                 temp_splits = [current_chunk[i:i + MAX_CHARS] for i in range(0, len(current_chunk), MAX_CHARS)]
                 text_chunks.extend(temp_splits)
            else:
                 text_chunks.append(current_chunk.strip())
        # -----------------------------------------------------------------

        total_chunks = len(text_chunks) 
        
        # キャッシュディレクトリが存在しない場合は作成
        os.makedirs(cache_dir, exist_ok=True) 

        for i, chunk in enumerate(text_chunks):
            # 1. キャッシュファイルのフルパスを決定
            wav_filename = self._get_cache_path(chunk, valence, intensity, narrator, i, total_chunks, cache_dir, base_name=base_filename, book_id=book_id, speed=speed)
            
            # 2. ファイルが存在する場合はスキップ（キャッシュヒット）
            if os.path.exists(wav_filename):
                 print(f"  チャンク {i+1}/{total_chunks} はキャッシュに存在します。合成をスキップ。")
                 continue
                 
            # 3. ファイルが存在しない場合は合成を実行
            try:
                # テキストクリーンアップ
                clean_chunk = re.sub(r'　+', ' ', chunk).strip() 
                clean_chunk = clean_chunk.replace('・・・・・・', '...').replace('・・・', '...')
                clean_chunk = re.sub(r'\s+', ' ', clean_chunk) 
                
                synth(
                    text=clean_chunk, # クリーンアップ後のテキストを使用
                    narrator=narrator, 
                    valence=valence, 
                    intensity=intensity, 
                    out_path=wav_filename,
                    speed=speed,
                )
                display_name = os.path.basename(wav_filename).split('__')[0]
                print(f"  チャンク {i+1}/{total_chunks} の合成完了 (ファイル名: {display_name}.wav)")
                
            except Exception as e:
                print(f"【エラー】音声合成失敗 (チャンク {i+1}): {e}")
                

        # --- 追加: 1ページ内で複数チャンクが生成された場合は wav を統合して1ファイルにする ---
        if total_chunks > 1 and base_filename:
            try:
                # 生成（またはキャッシュ）された各チャンクwavを順番に集める
                chunk_paths: typing.List[str] = []
                for ci, ctext in enumerate(text_chunks):
                    cp = self._get_cache_path(ctext, valence, intensity, narrator, ci, total_chunks, cache_dir,
                                             base_name=base_filename, book_id=book_id, speed=speed)
                    if not os.path.exists(cp):
                        raise FileNotFoundError(f"結合対象のチャンクファイルが見つかりません: {cp}")
                    chunk_paths.append(cp)

                # 統合後のwav（チャンク番号なし）
                merged_path = self._get_cache_path(text, valence, intensity, narrator, 0, 1, cache_dir,
                                                   base_name=base_filename, book_id=book_id, speed=speed)

                # すでに統合ファイルが存在する場合はスキップ
                if os.path.exists(merged_path):
                    print(f"  統合wavは既に存在します。結合をスキップ (ファイル: {os.path.basename(merged_path)})")
                else:
                    self._concat_wavs(chunk_paths, merged_path, silence_ms=120)
                    print(f"  ✅ チャンクwavを統合して保存しました (ファイル: {os.path.basename(merged_path)})")

                # 混在・重複再生を防ぐため、元チャンクを削除（必要ならコメントアウト）
                for cp in chunk_paths:
                    try:
                        os.remove(cp)
                    except Exception:
                        pass

            except Exception as e:
                print(f"【警告】チャンクwavの統合に失敗しました: {e}")

        print(f"--- プリロード終了 ---")
    
    # --- ヘルパー: wavファイルの結合（同一フォーマット前提） ---
    def _concat_wavs(self, wav_paths: typing.List[str], out_path: str, silence_ms: int = 120) -> None:
        """wav_paths を順番どおりに結合し、out_path に保存する（VOICEPEAK出力同士など同一フォーマット前提）。"""
        if not wav_paths:
            raise ValueError("wav_paths が空です。")

        # 1つ目を基準にフォーマットを決める
        with wave.open(wav_paths[0], "rb") as w0:
            nchannels = w0.getnchannels()
            sampwidth = w0.getsampwidth()
            framerate = w0.getframerate()
            comptype = w0.getcomptype()
            compname = w0.getcompname()

        silence_frames = int(framerate * (silence_ms / 1000.0))
        silence_bytes = b"\x00" * silence_frames * nchannels * sampwidth if silence_ms > 0 else b""

        with wave.open(out_path, "wb") as wout:
            wout.setnchannels(nchannels)
            wout.setsampwidth(sampwidth)
            wout.setframerate(framerate)
            wout.setcomptype(comptype, compname)

            for i, wp in enumerate(wav_paths):
                with wave.open(wp, "rb") as win:
                    fmt = (win.getnchannels(), win.getsampwidth(), win.getframerate(), win.getcomptype())
                    if fmt != (nchannels, sampwidth, framerate, comptype):
                        raise ValueError(
                            f"フォーマット不一致: {wp} expected(ch={nchannels}, sw={sampwidth}, fr={framerate}, ct={comptype}) "
                            f"actual(ch={win.getnchannels()}, sw={win.getsampwidth()}, fr={win.getframerate()}, ct={win.getcomptype()})"
                        )
                    wout.writeframes(win.readframes(win.getnframes()))

                if silence_bytes and i != len(wav_paths) - 1:
                    wout.writeframes(silence_bytes)

    # --- Sota通信メソッド ---

    def read_axes(self) -> dict: # ... (省略)
        with self.__connect() as conn:
            self.__send(conn, 'read_axes'.encode('utf-8'))
            data = self.__recv(conn)
        axes = json.loads(data)
        return axes

    def play_pose(self, pose: dict):
        data = json.dumps(pose).encode('utf-8')
        with self.__connect() as conn:
            self.__send(conn, 'play_pose'.encode('utf-8'))
            self.__send(conn, data)

    def stop_pose(self):
        with self.__connect() as conn:
            self.__send(conn, 'stop_pose'.encode('utf-8'))

    def play_motion(self, motion: typing.List[dict]):
        data = json.dumps(motion).encode('utf-8')
        with self.__connect() as conn:
            self.__send(conn, 'play_motion'.encode('utf-8'))
            self.__send(conn, data)

    def stop_motion(self):
        with self.__connect() as conn:
            self.__send(conn, 'stop_motion'.encode('utf-8'))

    def play_idle_motion(self, speed=1.0, pause=1000):
        data = json.dumps(dict(Speed=speed, Pause=pause)).encode('utf-8')
        with self.__connect() as conn:
            self.__send(conn, 'play_idle_motion'.encode('utf-8'))
            self.__send(conn, data)

    def stop_idle_motion(self):
        with self.__connect() as conn:
            self.__send(conn, 'stop_idle_motion'.encode('utf-8'))
            
    def play_wav_data(self, data: bytes):
        """WAVのバイト列を直接Sotaに送信する"""
        with self.__connect() as conn:
            self.__send(conn, 'play_wav'.encode('utf-8'))
            self.__send(conn, data)
            
    def play_wav_data_ack(self, data: bytes, duration_ms: int, timeout: float = 120.0) -> None:
        """
        プロトコル:
        - `PLAY\n`
        - 4byte big-endian: duration_ms
        - 4byte big-endian: wav size
        - wav bytes
        - server -> `ACK\n`
        """
        t0 = time.monotonic()
        print(f"▶ [ACK] send start: {len(data)} bytes, duration={duration_ms}ms to {self.__ip}:{self.__audio_port}")

        with self.__connect_audio() as conn:
            conn.settimeout(timeout)
            conn.sendall(b"PLAY\n")
            conn.sendall(int(duration_ms).to_bytes(4, "big", signed=True))
            conn.sendall(len(data).to_bytes(4, "big"))
            conn.sendall(data)

            # ACK行（\nまで）を待つ
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(1)
                if not chunk:
                    raise RuntimeError("connection closed before ACK")
                buf += chunk

        dt = time.monotonic() - t0
        if buf.strip() != b"ACK":
            raise RuntimeError(f"unexpected response: {buf!r}")
        print(f"✅ [ACK] received after {dt:.2f}s")


    def put_wav_cache(self, key: str, data: bytes, duration_ms: int, timeout: float = 120.0) -> None:
        """
        Sota上(AudioAckServer)に wav を保存だけしておく（再生しない）。

        プロトコル:
        - `PUT <key>\n`
        - 4byte big-endian: duration_ms（保存時のメタ情報。サーバ側でバリデーションに使う場合あり）
        - 4byte big-endian: wav size
        - wav bytes
        - server -> `OK\n`
        """
        t0 = time.monotonic()
        print(f"▶ [PUT] send start: key={key}, {len(data)} bytes, duration={duration_ms}ms to {self.__ip}:{self.__audio_port}")

        with self.__connect_audio() as conn:
            conn.settimeout(timeout)
            conn.sendall(f"PUT {key}\n".encode("utf-8"))
            conn.sendall(int(duration_ms).to_bytes(4, "big", signed=True))
            conn.sendall(len(data).to_bytes(4, "big"))
            conn.sendall(data)

            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(1)
                if not chunk:
                    raise RuntimeError("connection closed before OK")
                buf += chunk

        dt = time.monotonic() - t0
        if buf.strip() != b"OK":
            raise RuntimeError(f"unexpected response: {buf!r}")
        print(f"✅ [PUT] done after {dt:.2f}s (key={key})")

    def play_wav_key_ack(self, key: str, duration_ms: int, timeout: float = 300.0) -> None:
        """
        Sota上に保存済みの wav をキー指定で再生する（ACK待ち）。

        プロトコル:
        - `PLAYKEY <key>\n`
        - 4byte big-endian: duration_ms（待機時間に使う）
        - server -> `ACK\n`（再生完了後）
        """
        t0 = time.monotonic()
        print(f"▶ [PLAYKEY] send: key={key}, duration={duration_ms}ms to {self.__ip}:{self.__audio_port}")

        with self.__connect_audio() as conn:
            conn.settimeout(timeout)
            conn.sendall(f"PLAYKEY {key}\n".encode("utf-8"))
            conn.sendall(int(duration_ms).to_bytes(4, "big", signed=True))

            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(1)
                if not chunk:
                    raise RuntimeError("connection closed before ACK")
                buf += chunk

        dt = time.monotonic() - t0
        if buf.strip() != b"ACK":
            raise RuntimeError(f"unexpected response: {buf!r}")
        print(f"✅ [PLAYKEY] ACK after {dt:.2f}s (key={key})")


    # ===========================
    # 先読み（Sota側へ保存）関連
    # ===========================

    def preload_cached_speech_to_sota(
        self,
        base_filename: str,
        cache_dir: str,
        book_id: typing.Optional[str] = None,
        key_prefix: typing.Optional[str] = None,
        timeout: float = 120.0,
    ) -> typing.List[str]:
        """
        事前合成キャッシュwav（1ページ分）を Sota 上の AudioAckServer に「保存だけ」する。
        - 返り値: 保存したキーの順序リスト（再生順と一致）

        キー命名:
          key_prefix があればそれを使う（例: f"{book_id}_{base_filename}"）
          なければ (book_id と base_filename) から自動生成する。
          各wavに対して suffix "__{idx:03d}" を付ける。
        """
        if book_id is None:
            # book_id 省略時は prefix を base_filename のみにする
            book_id = ""
        chunk_files = self._get_cached_chunk_files(base_filename, cache_dir, book_id if book_id else None)
        if not chunk_files:
            print(f"【先読み】キャッシュが見つからないためスキップ: book_id={book_id}, base={base_filename}")
            return []

        if key_prefix is None:
            key_prefix = f"{book_id}_{base_filename}" if book_id else base_filename

        saved_keys: typing.List[str] = []
        for idx, wav_path in enumerate(chunk_files):
            # wavを読み込む
            with open(wav_path, "rb") as f:
                data = f.read()

            # 再生待機用の duration_ms を計算（Sota側はこれで待つ）
            duration_ms = self._calc_wav_duration_ms(wav_path)

            key = f"{key_prefix}__{idx:03d}"
            self.put_wav_cache(key=key, data=data, duration_ms=duration_ms, timeout=timeout)
            saved_keys.append(key)

        print(f"✅【先読み】Sotaへ保存完了: {key_prefix} ({len(saved_keys)} files)")
        return saved_keys

    def play_cached_speech_from_sota(
        self,
        key_prefix: str,
        cache_dir: str,
        base_filename: str,
        book_id: typing.Optional[str] = None,
        timeout: float = 300.0,
    ) -> float:
        """
        先読み済み（PUT済み）の音声を、Sota側の保存キーから再生する（送信なし）。
        - 返り値: 合計再生時間（秒）
        ※ キーの数を決めるために、ローカルの cache_dir から対応ファイルを列挙する。
        """
        if book_id is None:
            book_id = ""

        chunk_files = self._get_cached_chunk_files(base_filename, cache_dir, book_id if book_id else None)
        if not chunk_files:
            print(f"【再生】ローカルキャッシュが見つからない: {base_filename}")
            return 0.0

        total_sec = 0.0
        for idx, wav_path in enumerate(chunk_files):
            duration_ms = self._calc_wav_duration_ms(wav_path)
            key = f"{key_prefix}__{idx:03d}"
            self.play_wav_key_ack(key=key, duration_ms=duration_ms, timeout=timeout)
            total_sec += duration_ms / 1000.0

        return total_sec

    @staticmethod
    def _calc_wav_duration_ms(wav_path: str) -> int:
        """wavヘッダから再生時間(ms)を計算する（送信前/再生前の待機用）。"""
        try:
            import wave
            with wave.open(wav_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
            if rate <= 0:
                return 0
            return int((frames / rate) * 1000)
        except Exception:
            return 0

    def play_wav_batch_ack(
        self,
        items: typing.List[typing.Tuple[int, bytes]],
        timeout: float = 300.0
    ) -> None:
        """
        プロトコル(BATCH):
        - `BATCH\n`
        - 4byte big-endian: count
        - (count回繰り返し)
            - 4byte big-endian: duration_ms
            - 4byte big-endian: wav size
            - wav bytes
        - server -> `ACK\n`（全て再生完了後）
        """
        if not items:
            return

        t0 = time.monotonic()
        total_bytes = sum(len(b) for _, b in items)
        total_ms = sum(ms for ms, _ in items)
        print(f"▶ [ACK-BATCH] send start: files={len(items)}, total_bytes={total_bytes}, total_ms={total_ms} to {self.__ip}:{self.__audio_port}")

        with self.__connect_audio() as conn:
            conn.settimeout(timeout)

            conn.sendall(b"BATCH\n")
            conn.sendall(int(len(items)).to_bytes(4, "big", signed=False))

            for duration_ms, data in items:
                conn.sendall(int(duration_ms).to_bytes(4, "big", signed=True))
                conn.sendall(int(len(data)).to_bytes(4, "big", signed=False))
                conn.sendall(data)

            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(1)
                if not chunk:
                    raise RuntimeError("connection closed before ACK")
                buf += chunk

        dt = time.monotonic() - t0
        if buf.strip() != b"ACK":
            raise RuntimeError(f"unexpected response: {buf!r}")
        print(f"✅ [ACK-BATCH] received after {dt:.2f}s")


    def stop_wav(self):
        with self.__connect() as conn:
            self.__send(conn, 'stop_wav'.encode('utf-8'))

    def __send(self, conn: socket.socket, data: bytes):
        size = len(data).to_bytes(4, byteorder='big')
        conn.sendall(size)
        conn.sendall(data)

    def __connect(self) -> socket.socket:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((self.__ip, self.__port))
        return conn
    def __connect_audio(self) -> socket.socket:
        """ACK付き音声サーバ（audio_port）へ接続する。"""
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((self.__ip, self.__audio_port))
        return conn


    def __recvall(self, conn: socket.socket, size: int):
        chunks = []
        bytes_recved = 0
        while bytes_recved < size:
            chunk = conn.recv(size - bytes_recved)
            if chunk == b'':
                raise RuntimeError('socket connection broken')
            chunks.append(chunk)
            bytes_recved += len(chunk)
        return b''.join(chunks)

    def __recv_size(self, conn: socket.socket) -> int:
        b_size = self.__recvall(conn, 4)
        return int.from_bytes(b_size, byteorder='big')

    def __recv(self, conn: socket.socket) -> str:
        size = self.__recv_size(conn)
        return self.__recvall(conn, size).decode('utf-8')
    
    def __choose(self, prev: dict) -> dict:
        BEAT_ARM_SERVO_MAP_LIST = [
            {'LeftElbow': 10, 'RightElbow': -10},
            {'LeftElbow': -10, 'RightElbow': 10},
            {'LeftElbow': 20, 'RightElbow': -20},
            {'LeftElbow': -20, 'RightElbow': 20},
            {'LeftElbow': 0, 'RightElbow': 0}
        ]
        DEFAULT_ARM_SERVO_MAP = {'LeftElbow': 0, 'RightElbow': 0}
        while True:
            map = random.choice(BEAT_ARM_SERVO_MAP_LIST)
            if map != prev:
                return map
