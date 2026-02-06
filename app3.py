# app3.py
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
import os
import json

# ==============================================================================
# 1. 絵本データ定義 (Flask用) ★★★ 追加箇所 ★★★
# ==============================================================================
BOOK_DEFINITIONS_FOR_FLASK = {
    "ookinakabu": {
        "image_dir": "images/おおきなかぶ", # static/からのパス
        # page2.jpgからpage17.jpgまで、コンテンツページの画像ファイル名リスト
        "content_pages": [
            "page2.jpg", "page3.jpg", "page4.jpg", "page5.jpg", "page6.jpg", 
            "page7.jpg", "page8.jpg", "page9.jpg", "page10.jpg", "page11.jpg", 
            "page12.jpg", "page13.jpg", "page14.jpg", "page15.jpg", "page16.jpg", "page17.jpg"
        ],
        "last_page_image": "page18.jpg" # 最終ページ画像
    },
    "ichibansencho": {
        "image_dir": "images/いちばんせんちょう", # static/からのパス
        # page2.jpgからpage11.jpgまで、コンテンツページの画像ファイル名リスト (10枚)
        "content_pages": [
            "page2.jpg", "page3.jpg", "page4.jpg", "page5.jpg", "page6.jpg",
            "page7.jpg", "page8.jpg", "page9.jpg", "page10.jpg", "page11.jpg",
            "page12.jpg", "page13.jpg", "page14.jpg", "page15.jpg", "page16.jpg",
            "page17.jpg", "page18.jpg", "page19.jpg", "page20.jpg", "page21.jpg"
        ],
        "last_page_image": "page22.jpg" # 最終ページのファイル名を指定
    },
    "kanachan": {
        "image_dir": "images/かなちゃん", # static/からのパス
        # page2.jpgからpage11.jpgまで、コンテンツページの画像ファイル名リスト (10枚)
        "content_pages": [
            "page2.jpg", "page3.jpg", "page4.jpg", "page5.jpg", "page6.jpg",
            "page7.jpg", "page8.jpg", "page9.jpg", "page10.jpg", "page11.jpg",
            "page12.jpg", "page13.jpg", "page14.jpg", "page15.jpg", "page16.jpg",
            "page17.jpg", "page18.jpg", "page19.jpg", "page20.jpg", "page21.jpg",
            "page22.jpg","page23.jpg"
        ],
        "last_page_image": "page24.jpg" # 最終ページのファイル名を指定
    },
    "suhu": {
        "image_dir": "images/スーフと白い馬", # static/からのパス
        # page2.jpgからpage11.jpgまで、コンテンツページの画像ファイル名リスト (10枚)
        "content_pages": [
            "page2.png", "page3.png", "page4.png", "page5.png", "page6.png",
            "page7.png", "page8.png", "page9.png", "page10.png", "page11.png",
            "page12.png", "page13.png", "page14.png", "page15.png", "page16.png",
            "page17.png", "page18.png", "page19.png", "page20.png", "page21.png",
            "page22.png", "page23.png", "page24.png", "page25.png", "page26.png",
            "page27.png", "page28.png", "page29.png", "page30.png", "page31.png",
            "page32.png", "page33.png", "page34.png", "page35.png", "page36.png",
            "page37.png", "page38.png"
        ],
        "last_page_image": "page39.png" # 最終ページのファイル名を指定
    },
    "inu": {
        "image_dir": "images/犬", # static/からのパス
        # page2.jpgからpage11.jpgまで、コンテンツページの画像ファイル名リスト (10枚)
        "content_pages": [
            "page2.jpeg", "page3.jpeg", "page4.jpeg", "page5.jpeg", "page6.jpeg",
            "page7.jpeg", "page8.jpeg", "page9.jpeg", "page10.jpeg", "page11.jpeg",
            "page12.jpeg", "page13.jpeg", "page14.jpeg", "page15.jpeg", "page16.jpeg",
            "page17.jpeg", "page18.jpeg", "page19.jpeg", "page20.jpeg", "page21.jpeg",
            "page22.jpeg", "page23.jpeg", "page24.jpeg", "page25.jpeg", "page26.jpeg",
            "page27.jpeg", "page28.jpeg", "page29.jpeg", "page30.jpeg", "page31.jpeg",
            "page32.jpeg"
        ],
        "last_page_image": "page33.jpeg" # 最終ページのファイル名を指定
    }
}
# ★★★ 現在使用する絵本IDを指定 ★★★ (他のスクリプトと同期させてください)
CURRENT_FLASK_BOOK_ID = "suhu" 
# ==============================================================================


app = Flask(__name__)
# 本番環境ではより複雑なものに変更してください
app.config['SECRET_KEY'] = 'your_very_secret_key_for_socketio' 
socketio = SocketIO(app)

# Sotaの読み上げ完了を通知するエンドポイント
@app.route('/sota_reading_finished', methods=['POST'])
def sota_reading_finished():
    """
    SotaのPythonスクリプトから、読み上げ完了の通知を受け取るエンドポイント。
    受け取った通知をコンソールに出力し、WebSocketでページめくり指示を送信する。
    """
    if request.method == 'POST':
        data = request.get_json() 
        flip_duration = data.get('flip_duration', 600) if data else 600 

        print(f"Sotaから読み上げ完了通知を受信しました！")
        print(f"受信しためくり速度: {flip_duration}ms")
        
        # WebSocketを通じて、接続している全てのクライアントに'turn_page_command'を送信
        socketio.emit('turn_page_command', {
            'status': 'page_turned', 
            'message': 'Sota finished reading. Turning page...',
            'flip_duration': flip_duration # ここでめくり速度を送信
        })
        print("ウェブブラウザにページめくりコマンドを送信しました。")
        
        return "Notification Received", 200 
    return "Method Not Allowed", 405

# ウェブページを表示するルート
@app.route('/')
def index():
    # ★★★ 修正箇所: IDから設定をロードし、テンプレートに渡す ★★★
    book_config = BOOK_DEFINITIONS_FOR_FLASK.get(CURRENT_FLASK_BOOK_ID, BOOK_DEFINITIONS_FOR_FLASK["ookinakabu"])

    return render_template('index.html',
                           book_image_dir=book_config["image_dir"],
                           content_pages=book_config["content_pages"],
                           last_page_image=book_config["last_page_image"])
    # ★★★ 修正ここまで ★★★

# 読み聞かせ終了（アンケート表示）を通知するエンドポイント
@app.route('/reading_finished', methods=['POST'])
def reading_finished():
    data = request.get_json(silent=True) or {}
    survey_url = data.get('survey_url', '')

    print("📩 読み聞かせ終了通知を受信しました。アンケートQRを表示します。")
    socketio.emit('show_survey_qr', {'survey_url': survey_url})
    return jsonify({'ok': True}), 200


if __name__ == '__main__':
    # Flaskサーバーを起動
    socketio.run(app, debug=True)