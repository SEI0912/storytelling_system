import java.io.*;
import java.net.*;
import java.nio.file.*;

import jp.vstone.RobotLib.CPlayWave;
import jp.vstone.RobotLib.CRobotUtil;

public class AudioAckServer {

    // ===== 設定 =====
    private static final int PORT = 30001;
    private static final long SAFETY_MS = 0; // 必要なら 600〜1200 など

    // 保存先（RAMディスク推奨）
    private static final String CACHE_DIR = "/dev/shm/tts_cache";
    private static final String TMP_PLAY_PATH = "/dev/shm/tts.wav";

    // 再生は同時に走らせない（ファイル上書き＆音が混ざるのを防ぐ）
    private static final Object playLock = new Object();

    // BufferedReaderを使わずに、バイトで1行読む（\nまで）
    private static String readLineAscii(InputStream in, int maxLen) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        for (int i = 0; i < maxLen; i++) {
            int b = in.read();
            if (b == -1) break;
            if (b == '\n') break;
            if (b != '\r') bos.write(b);
        }
        if (bos.size() == 0) return null;
        return bos.toString("UTF-8");
    }

    private static void ensureCacheDir() throws IOException {
        Files.createDirectories(Paths.get(CACHE_DIR));
    }

    private static String safeKeyToPath(String key) {
        // パス注入防止：英数と _- . のみに制限
        String safe = key.replaceAll("[^A-Za-z0-9_\\-\\.]", "_");
        return CACHE_DIR + "/" + safe + ".wav";
    }

    private static void playBytes(byte[] wav, int durationMs) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(TMP_PLAY_PATH)) {
            fos.write(wav);
        }
        CPlayWave.PlayWave(TMP_PLAY_PATH);

        long minWait = 300; // durationMsが0でも即ACKしない保険
        long waitMs = Math.max(minWait, (long) durationMs + SAFETY_MS);
        CRobotUtil.wait((int) Math.min(Integer.MAX_VALUE, waitMs));
    }

    private static void handleClient(Socket s) {
        try {
            s.setTcpNoDelay(true);

            InputStream rawIn = new BufferedInputStream(s.getInputStream());
            OutputStream out = new BufferedOutputStream(s.getOutputStream());

            // 先頭行：コマンド
            String line = readLineAscii(rawIn, 256);
            if (line == null) return;

            String[] parts = line.trim().split("\\s+");
            String cmd = parts[0];

            DataInputStream din = new DataInputStream(rawIn);

            // PUT key : wavをキャッシュに保存（再生しない）
            if ("PUT".equals(cmd)) {
                if (parts.length < 2) {
                    out.write("ERR\n".getBytes("UTF-8"));
                    out.flush();
                    return;
                }
                ensureCacheDir();

                String key = parts[1];
                int durationMs = din.readInt(); // big-endian
                int n = din.readInt();          // big-endian

                if (n <= 0 || n > 50_000_000 || durationMs < 0 || durationMs > 600_000) {
                    out.write("ERR\n".getBytes("UTF-8"));
                    out.flush();
                    return;
                }

                byte[] wav = new byte[n];
                din.readFully(wav);

                String path = safeKeyToPath(key);
                try (FileOutputStream fos = new FileOutputStream(path)) {
                    fos.write(wav);
                }

                out.write("OK\n".getBytes("UTF-8"));
                out.flush();
                return;
            }

            // PLAYKEY key : キャッシュ済みwavを再生
            if ("PLAYKEY".equals(cmd)) {
                if (parts.length < 2) {
                    out.write("ERR\n".getBytes("UTF-8"));
                    out.flush();
                    return;
                }
                String key = parts[1];
                int durationMs = din.readInt();
                String path = safeKeyToPath(key);

                if (!Files.exists(Paths.get(path))) {
                    out.write("NOFILE\n".getBytes("UTF-8"));
                    out.flush();
                    return;
                }

                byte[] wav = Files.readAllBytes(Paths.get(path));

                synchronized (playLock) {
                    playBytes(wav, durationMs);
                }

                out.write("ACK\n".getBytes("UTF-8"));
                out.flush();
                return;
            }

            // PLAY : その場で受け取ったwavを再生（互換用）
            if ("PLAY".equals(cmd)) {
                int durationMs = din.readInt();
                int n = din.readInt();

                if (n <= 0 || n > 50_000_000 || durationMs < 0 || durationMs > 600_000) {
                    out.write("ERR\n".getBytes("UTF-8"));
                    out.flush();
                    return;
                }

                byte[] wav = new byte[n];
                din.readFully(wav);

                synchronized (playLock) {
                    playBytes(wav, durationMs);
                }

                out.write("ACK\n".getBytes("UTF-8"));
                out.flush();
                return;
            }

            // BATCH : その場で複数wavを連続再生（互換用）
            if ("BATCH".equals(cmd)) {
                int count = din.readInt();
                if (count <= 0 || count > 200) {
                    out.write("ERR\n".getBytes("UTF-8"));
                    out.flush();
                    return;
                }

                synchronized (playLock) {
                    for (int i = 0; i < count; i++) {
                        int durationMs = din.readInt();
                        int n = din.readInt();

                        if (n <= 0 || n > 50_000_000 || durationMs < 0 || durationMs > 600_000) {
                            out.write("ERR\n".getBytes("UTF-8"));
                            out.flush();
                            return;
                        }

                        byte[] wav = new byte[n];
                        din.readFully(wav);

                        playBytes(wav, durationMs);
                    }
                }

                out.write("ACK\n".getBytes("UTF-8"));
                out.flush();
                return;
            }

            // 不明コマンド
            out.write("ERR\n".getBytes("UTF-8"));
            out.flush();

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try { s.close(); } catch (Exception ignore) {}
        }
    }

    public static void main(String[] args) throws Exception {
        ServerSocket ss = new ServerSocket(PORT);
        System.out.println("AudioAckServer listening on " + PORT);

        while (true) {
            Socket s = ss.accept();
            // 接続ごとに別スレッドで処理（PLAY中でもPUT可能）
            new Thread(() -> handleClient(s)).start();
        }
    }
}
