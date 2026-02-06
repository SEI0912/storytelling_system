# voicepeak_cli_min.py
import subprocess, os, tempfile, shutil

VP = "/Applications/voicepeak.app/Contents/MacOS/voicepeak"

def synth(text, narrator, valence=0.0, intensity=0.0, speed=100, out_path="out.wav"):
    # 1. V, I をクリップ
    v = max(-1.0, min(1.0, float(valence)))   # -1〜+1
    I = max(-1.0, min(1.0, float(intensity))) # -1〜+1（今は未使用）

    # 2. 正負バレンス & 強度
    v_pos = max(v, 0.0)     # ポジティブ側の大きさ [0,1]
    v_neg = max(-v, 0.0)    # ネガティブ側の大きさ [0,1]
    I_mag = abs(I)          # 強度の大きさ [0,1]（必要なら後で使えるように残しておく）

    # 3. V項のモデル（happy / sad）
    alpha_v = 0.7  # V（極性）の寄与

    if v > 0:
        happy_norm = alpha_v * v_pos
        sad_norm   = 0.0
    elif v < 0:
        sad_norm   = alpha_v * v_neg
        happy_norm = 0.0
    else:
        # V ≒ 0 のときはニュートラル
        happy_norm = 0.0
        sad_norm   = 0.0
        # 「強いけど中立」の声にしたいならここで I_mag を使う

    # 4. 0〜1 にクリップ
    happy_norm = max(0.0, min(1.0, happy_norm))
    sad_norm   = max(0.0, min(1.0, sad_norm))

    # 5. Voicepeak の 0〜100 にスケーリング
    happy = round(100 * happy_norm)
    sad   = round(100 * sad_norm)

    emo = f"happy={happy},sad={sad}"

    # 6. valence からピッチ[%]を計算（0%が平均、-150〜+150%）
    PITCH_MAX = 150.0  # [%]
    pitch_percent = v * PITCH_MAX
    pitch_int = int(round(pitch_percent))  # VOICEPEAK に渡す整数

    speed_int = max(50, min(150, int(speed)))

    tmpdir = tempfile.mkdtemp(prefix="vp_")
    try:
        tmp = os.path.join(tmpdir, "seg.wav")
        cmd = [VP, "-s", text, "-o", tmp, "-n", narrator, "-e", emo, "--speed", str(speed_int)]#, "--speed", str(speed_int)

        # ピッチ指定を追加
        cmd.extend(["--pitch", str(pitch_int)])

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(tmp):
            raise RuntimeError(
                f"VOICEPEAK error\ncmd:{' '.join(cmd)}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )

        shutil.copy2(tmp, out_path)

        # ファイルパスと感情パラメータ & pitch を返却
        return {
            "out_path": out_path,
            "happy": happy,
            "sad": sad,
            "pitch": pitch_int,
            "speed": speed_int  # 実際に使われたピッチ[%]
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
