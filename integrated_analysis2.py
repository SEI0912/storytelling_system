import os
import json
import math
import numpy as np
from typing import List, Dict, Union, Optional, Any

from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import warnings

from google.cloud import language_v1
from google.api_core.client_options import ClientOptions

import openai


# ==============================================================================
# 0. ユーティリティ：符号保持のバッチ正規化（-1〜+1） / 0〜1正規化
# ==============================================================================
def normalize_signed(values: List[float]) -> List[float]:
    """符号を保持したまま -1〜+1 にバッチ正規化（負側と正側を別レンジでスケール）。"""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    max_abs = max(abs(vmin), abs(vmax)) or 1e-6
    out = []
    for v in values:
        if v >= 0:
            denom = vmax if vmax > 0 else max_abs
            out.append(v / (denom if abs(denom) > 1e-12 else 1.0))
        else:
            denom = abs(vmin) if vmin < 0 else max_abs
            out.append(v / (denom if abs(denom) > 1e-12 else 1.0))
    return [max(-1.0, min(1.0, x)) for x in out]


def normalize_01(values: List[float], fallback: float = 0.5) -> List[float]:
    """0〜1 にバッチ正規化。全値同一時は fallback を返す。"""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-12:
        return [fallback for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def head(text: str, n: int = 100) -> str:
    """次ページテキストを渡しすぎない用（先読み抑制）"""
    return (text or "")[:n]


# ==============================================================================
# 0.5. 📚 絵本データ定義（ここはあなたの元コードをそのまま）
# ==============================================================================
BOOK_DEFINITIONS = {
    "ookinakabu": {
        "image_dir": "static/images/おおきなかぶ感情分析用/",
        "pages": [
            {"text": "おじいさんがかぶを植えました。大きくなあれ、甘くなあれ、でっかいかぶになっとくれ。", "image_path_suffix": "page2.png"},
            {"text": "かぶはでっかくなりました。さあぬこう、うんこらしょ、どっこいしょ。かぶはびくとも動きません。ばあさん、ばあさん、来ておくれ。", "image_path_suffix": "page3.png"},
            {"text": "はいはい私も手伝いましょ。2人で引っ張ろいっせのせ、うんこらしょ、どっこいしょ。やっぱりかぶは抜けません。今度は誰を呼ぼうかな。孫のマーシャ、来ておくれ。", "image_path_suffix": "page4.png"},
            {"text": "はいはい私も手伝うわ。みんなでひっぱろいっせのせ、うんこらしょ、どっっこいしょ。やっぱりかぶは抜けません。今度は誰を呼ぼうかな犬のポーチ、来ておくれ。", "image_path_suffix": "page5.png"},
            {"text": "ワンワン僕も手伝うワン。みんなでひっぱろいっせのせ。うんこらしょ、どっこいしょ。ワンワンワン。やっぱりかぶは抜けません。今度は誰を呼ぼうかな。ねこのニャーゴ、来ておくれ。ニャアニャア私も手伝うニャア。", "image_path_suffix": "page6.png"},
            {"text": "みんなでひっぱろいっせのせ。うんこらしょ、どっこいしょ、ワンワンワン、ニャアニャアニャア。やっぱりかぶは抜けません。今度は誰を呼ぼうかな。ねずみのチュータ、来ておくれ。チュウチュウチュウ、僕も手伝うチュウ。みんなでひっぱろいっせのせ、うんこらしょ、どっこいしょ。", "image_path_suffix": "page7.png"},
            {"text": "ずっぽーん！やっとかぶは抜けました。", "image_path_suffix": "page8.png"},
            {"text": "大きな株持って帰ろ。よいしょよいしょ、ワンワン、ニャアニャア、チュウチュウチュウ。みんなで食べよおいしいよ。いただきまあす。ああ美味しい、ああ美味しい。みんなで抜いた甘いかぶ。", "image_path_suffix": "page9.png"}
        ]
    },
    "ichibansencho": {
        "image_dir": "static/images/いちばんせんちょう感情分析用/",
        "pages": [
            {"text": "いちくんが白い帽子をかぶって、いちばんせんちょう、いちばんせんとう、ぼくのうしろをついてこい。歌いながら歩いていたら、", "image_path_suffix": "page2.jpg"},
            {"text": "野原に出ました。「どっちへ行こうかな。」いちくんがまよっていると、うさぎが草むらからぴょこんと顔を出して、「こっちがいいよ」と言いました。", "image_path_suffix": "page3.jpg"},
            {"text": "うさぎはいちくんの帽子をかぶって歌います。いちばんせんちょう、いちばんせんとう、僕の後ろをついてこい。", "image_path_suffix": "page4.jpg"},
            {"text": "うさぎのあとをついていったら、小さな川がありました。「どっちへ行こうか。」いちくんとうさぎが話していると、川で泳いでいたあひるが、「こっちがいいよ。」といいました。", "image_path_suffix": "page5.jpg"},
            {"text": "あひるはいちくんの帽子をかぶって歌います。いちばんせんちょう、いちばんせんとう、僕の後ろをついてこい。", "image_path_suffix": "page6.jpg"},
            {"text": "アヒルのあとを、うさぎといちくんがついていったら、そこは森の中。ちょっと暗くて怖かったので、「どうしよう」「どっちへ行こうか」と話していると、クマが木の陰からのっそり出てきて言いました。「森から出るならこっちだよ」", "image_path_suffix": "page7.jpg"},
            {"text": "くまはいちくんの帽子をかぶって歌います。いちばんせんちょう、いちばんせんとう、僕の後ろをついてこい。", "image_path_suffix": "page8.jpg"},
            {"text": "クマのあとをついて森を出たら、あれあれ、そこはいちくんのうち。母さんがにっこり笑っています。", "image_path_suffix": "page9.jpg"},
            {"text": "いちばんせんちょう、いちばんせんとう、みんないっしょについてこい。みんなでついていったら・・・", "image_path_suffix": "page10.jpg"},
            {"text": "おやつの用意がしてありました。いちばんせんちょう、いちばんせんとう、みんなでいっしょにいただきます。", "image_path_suffix": "page11.jpg"}
        ]
    },
    "kanachan": {
        "image_dir": "static/images/かなちゃん感情分析用/",
        "pages": [
            {"text": "ブルくんの耳にちょうちょがとまりました。「かなちゃん、みてみて！かなちゃんのリボンとお揃いだよ。」でも、かなちゃんはありの行列に夢中です。「ありんこくん、どこまでいくのかな？」", "image_path_suffix": "page2.jpg"},
            {"text": "ボスン！かなちゃんは、蟻を追いかけて、木の中に潜り込みました。ガサガサ、「あっちから出てきた！」", "image_path_suffix": "page3.jpg"},
            {"text": "かなちゃんはありを追いかけて、窓から部屋の中へ入って行きました。「クゥ〜ン」ブルくんは入れません。", "image_path_suffix": "page4.jpg"},
            {"text": "「そうか！玄関から入ればいいんだ」ブルくんは、急いで家のなかへ入って行きました。", "image_path_suffix": "page5.jpg"},
            {"text": "「みて、ブルくん！ありんこくんがクッキーのかけらに集まっているよ！」ブルくんが見てみると、部屋の中もありの行列でいっぱいです。", "image_path_suffix": "page6.jpg"},
            {"text": "すると、とつぜんかなちゃんが言いました。「あれ？クマちゃんがいない！リボンもないよ」そして、「うわ〜ん！」と泣きだしてしまいました。", "image_path_suffix": "page7.jpg"},
            {"text": "ブルくんは、ありの行列を戻ってみました。「かなちゃん！くまがいたよ！」もっと戻ってみました。「かなちゃん！リボンがあったよ！」", "image_path_suffix": "page8.jpg"},
            {"text": "ブルくんはかなちゃんに駆け寄りました。「かなちゃん、リボン持ってきたよ！」でも、かなちゃんはぬいぐるみをぎゅっと抱きしめると、言いました。「よかった、ここにいた！クマちゃん、大好き！」", "image_path_suffix": "page9.jpg"},
            {"text": "そしてかなちゃんは、「ママー！」というと、向こうへ行ってしまいました。", "image_path_suffix": "page10.jpg"},
            {"text": "「かなちゃんは、ぼくよりありんこよりクマちゃんが大好きなんだ」ブルくんがしょんぼりしていると・・・", "image_path_suffix": "page11.jpg"},
            {"text": "かなちゃんが戻ってきました。「ブルくん！ブルくんの大好きなドーナッツ、ママからもらってきたよ。さっきはありがとうね！」", "image_path_suffix": "page12.jpg"}
        ]
    },
    "suhu": {
        "image_dir": "static/images/スーフと白い馬感情分析用/",
        "pages": [
            {"text": "ずっとずっと昔、モンゴルの草原に、スーフという貧しい羊飼いの少年が、としとったおばあさんと2人で暮らしていました。  優しくて働き者のスーフは、毎日おばあさんを助けてよく働きました。", "image_path_suffix": "page2.png"},
            {"text": "スーフの仕事は、飼っている大切の羊を、毎日、美味しい草のある草原まで連れて行くことでした。  スーフは歌を歌いながら、羊を追っていきます。  スーフの歌声は、風に乗って草原を流れていきます。  羊たちも草原で暮らす人たちも、スーフの歌声にうっとりするのでした。", "image_path_suffix": "page3.png"},
            {"text": "ある日のこと、夕暮れになってもスーフは帰ってきません。  おばあさんは、心配で心配でなりません。  そうしていると、遠くから、白いものをだきかかえたスーフが帰ってきました。", "image_path_suffix": "page4.png"},
            {"text": "よくみると、それは生まれたばかりの白い子馬でした。  「帰る途中で見つけたんだ。  ひとりぼっちで倒れていたんだ。  おばあちゃん、この子馬、飼ってもいいでしょう？  僕が世話をするから・・・・・」", "image_path_suffix": "page5.png"},
            {"text": "その日から、スーフは一生懸命、小馬の世話をしました。  夜眠る時も、スーフは子馬と一緒です。  兄弟のいないスーフに、弟か妹ができたみたいです。  スーフが歌えば、小馬も歌います。  スーフのあとを、どこへでもとことこついていく子馬の姿を見ると、草原の誰もが微笑ましく思うのでした。  そして、子馬はどんどん大きくなり、雪のように白い美しい馬になっていきました。", "image_path_suffix": "page6.png"},
            {"text": "あるとし、この国の王様が、競馬の大会をすることになりました。  一等になったものを、娘と結婚させるというのです。  スーフも白い馬と参加することになりました。  競馬大会の場所で、何千という立派な馬が集まっていました。", "image_path_suffix": "page7.png"},
            {"text": "「よーい、どん！！」  たくましい男たちがムチを振ると、何千という馬が、いっせいに走り出しました。", "image_path_suffix": "page8.png"},
            {"text": "初めは後ろの方を走っていた白い馬が、どんどん他の馬を、追い抜いて・・・・・・、追い抜いて・・・・・・、先頭をかけていきます。  白い立て髪をなびかせ、風のように駆け抜けていきます。  見物人の歓声が、草原じゅうに響き渡ります。", "image_path_suffix": "page9.png"},
            {"text": "王様が叫びました。  「いっとうの白い馬と、乗り手を捕まえてまいれー！」", "image_path_suffix": "page10.png"},
            {"text": "王様は、スーホの姿を見るなり言いました。  「お前みたいなみすぼらしい羊飼いに、娘はやれん！  銀貨を3枚やるから、その白い馬を置いて、とっとと帰れ!」  王様は3枚の銀貨をスーフに投げつけました。  スーフは言いました。  「私は競馬に来たのです。  馬を売りに来たのではありません。  この白い馬は私にとって、大切な馬なのです」  「生意気なやつめ！こいつを痛い目に合わせろ！」", "image_path_suffix": "page11.png"},
            {"text": "大勢の家来に殴られ、スーフはとうとう倒れてしまいました。  そして、王様は白い馬を奪い取っていきました。", "image_path_suffix": "page12.png"},
            {"text": "仲間の羊飼いがスーフを担いで、おばあさんのうちまで連れて行ってくれました。  おばあさんの看病のおかげで、スーフの傷はだんだん治っていきましたが、白い馬を奪われた悲しみは深くなるばかりでした。", "image_path_suffix": "page13.png"},
            {"text": "そのころ王様は、白い馬を自慢する会を開いていました。  王様が白い馬に乗った瞬間、白い馬は突然暴れ出し、王様を振り落としました。  カンカンに怒った王様は怒鳴りました。  「殺せ！殺せ！あいつを殺せーー！」", "image_path_suffix": "page14.png"},
            {"text": "家来たちが放った矢が、次々に白い馬に刺さりました。  それでも白い馬は走りました。  スーフに会いたくて会いたくて走り続けました。", "image_path_suffix": "page15.png"},
            {"text": "走って走って、白い馬はやっとスーフのところに帰ってきました。体には何本もの矢が突き刺さり、白い馬の体は、真っ赤に染まっていました。  「帰ってきたんだね。  こんなになりながら・・・、僕に会いにきてくれたんだね」  スーフは泣きながら矢を抜き取りました。  けれども白い馬は大好きなスーフの腕の中で、静かに息を引き取っていきました。", "image_path_suffix": "page16.png"},
            {"text": "スーフは毎日泣いていました。  ある夜、スーフの夢の中に、白い馬が現れ、こう言いました。  「もう泣かないでください！  お願いがあります。  私の体で楽器を作ってください。そうすれば、私はあなたといつも一緒にいられます。  あなたが歌う時私も一緒に歌えます。」", "image_path_suffix": "page17.png"},
            {"text": "白い馬に言われた通り、スーフはすぐに楽器を作り始めました。  何日も何日もかかって、とうとう楽器が出来上がりました。  何から何まで、白い馬の体でできています。", "image_path_suffix": "page18.png"},
            {"text": "来る日も来る日も、スーフは楽器をひき続けました。  それをひくとスーフは、白い馬と一緒にいるような気がしました。  そして、「ヒヒィーン！」という音色も、「パカパカ・・・」という音色も、ひけるようになったのです。  それは、白い馬の「いななき」そっくりです。  それは、白い馬の「ひづめの音」そっくりです。  まるで、白い馬がここにいるようです。", "image_path_suffix": "page19.png"},
            {"text": "スーフのひく楽器の音色が、モンゴルの草原を流れていきます。  そうして美しい音色は、草原に暮らすすべての人々、すべての動物たちの心を慰め、励ましてくれるのでした。  これがモンゴルに伝わる、楽器「馬頭琴」のお話です。", "image_path_suffix": "page20.png"}
        ]
    },
    "inu": {
        "image_dir": "static/images/犬感情分析用/",
        "pages": [
            {"text": "バイオリンひきのヘクターが、街で演奏しています。  聞いているのは、犬のヒューゴ。  ふたりは大の仲良しです。  ヒューゴは、世界の誰よりもヘクターのひくバイオリンが好きでした。  楽しい時も、悲しい時も、いつも一緒に過ごしてきました。", "image_path_suffix": "page2.jpg"},
            {"text": "でもある晩のこと、トボトボと家に向かいながら、ヘクターは言いました。  「ヒューゴや、これから私はどうしたらいいのだろう。昨日のニュースを見たかい？世界的に有名なくまのピアニストが活躍しているそうじゃないか。そんなときに、誰が、こんな老ぼれのバイオリンを聴きたいと思うかね」  「僕は聞きますよ、喜んで！」と言うように、ヒューゴはクウウンと鼻を鳴らしました。  でも、ヘクターはため息をつくばかり。  「私は、チャレンジするには、歳をとりすぎた。大きなコンサート会場で演奏するなんて、夢のまた夢だ」  そう言って、バイオリンをしまい、二度と手に取ろうとはしませんでした。", "image_path_suffix": "page3.jpg"},
            {"text": "演奏をしなくなったヘクターは、テレビを見て1日を過ごします。  ときには、音楽を聴き、ときには、居眠りをします。  さらに居眠りをして、またまた居眠りです。", "image_path_suffix": "page4.jpg"},
            {"text": "ヘクターとヒューゴが住んでいたのは、結構騒がしいところでしたから、眠るときには必ず窓を閉めます。  でも、ある晩、うっかり閉め忘れてしまいました。  真夜中、不思議な音で、ヘクターは目を覚ましました。  ベッドから抜け出して・・・  廊下を、そっと歩き・・・. 屋上に続くドアを開けてみると・・・", "image_path_suffix": "page5.jpg"},
            {"text": "犬のヒューゴが、ヘクターのバイオリンを弾いていました。  その演奏は、踊り出したくなるような、手を叩きたくなるような、一緒に歌いたくなるような・・・つまり、素晴らしいものでした！  みんながうっとりしているのを見て、ヘクターは心のどこかがキュッと痛みました。  「自分の友達が、バイオリンの天才だったとは！」", "image_path_suffix": "page6.jpg"},
            {"text": "次の日の朝、ヘクターは出来る限りの技を、全てヒューゴに教えました。  ヒューゴがバイオリンを弾き始めると、瞬く間に人だかりです。", "image_path_suffix": "page7.jpg"},
            {"text": "素晴らしい演奏をする犬のニュースは、あっという間に広がりました。そしてある日、あの有名なクマのピアニスト、ブラウンが聴きにきたのです。  ブラウンはヒューゴに言いました。  「動物だけの楽団を始めようと思うんだ。名前はブラウン楽団。一緒に世界ツアーに行かないか？何百人、いや、何千人の前で、君にバイオリンを弾いてほしい。」  ヒューゴが尻尾を振りながら嬉しそうにヘクターを見たとき、ヘクターの心はまたキュッと痛みました。「行ったらいいさ。一生に一度のチャンスじゃないか」  ぎこちなく微笑みながら、ヘクターは言いました。", "image_path_suffix": "page8.jpg"},
            {"text": "ブラウン楽団と出発するために、ヒューゴは準備をしています。楽しそうに尻尾を振りながら、バイオリンを大切にしまいます。  ところが、ヘクターは、こんなことを言い出しました。「ヒューゴ、君はあんなつまらない奴らと演奏するのかい？ちっとも、いいこととは思えないんだが」  ヒューゴはお願いするように、ヘクターを見上げました。でもヘクターは、いい顔をしてくれません。  「いいだろう。いけばいいさ。でもきっと、尻尾を巻いて帰ってくるに違いない。君は、そこまで上手くないからね！」  ヒューゴは背を向け、スーツケースを持って出て行きました。ヘクターは急に寂しくなって、叫びました。「待ってくれ、ヒューゴ！ごめ・・・」", "image_path_suffix": "page9.jpg"},
            {"text": "でも、その声は届きませんでした。", "image_path_suffix": "page10.jpg"},
            {"text": "ブラウン楽団と一緒に、ヒューゴは世界中を回りました。チケットはいつも売り切れ。大勢の観客を前に、素晴らしい演奏を続けました。", "image_path_suffix": "page11.jpg"},
            {"text": "今やヒューゴは大スターです。クマのブラウンがピアノ、きりんのビッグジラフがドラム、そしてコントラバスは、オオカミのウルフマン。世界中の人々が、この楽団の演奏をテレビやパソコンでも楽しみました。その中にはヘクターもいます。演奏を見ながら、ヘクターは自分の音楽を懐かしく思い出しました。バイオリンを弾いていた頃が恋しくなりました。でも一番恋しいのは友達のヒューゴです。", "image_path_suffix": "page12.jpg"},
            {"text": "数ヶ月経ったある日のこと、ヘクターはブラウン楽団のポスターを目にしました。彼らは、この街1番のコンサート会場で演奏をしているのです。行きたい、とヘクターは思いました。でも同時に、ヒューゴに酷いことを言ってしまったのを思い出しました。ヒューゴは私の顔なんか見たくないんじゃないだろうか？", "image_path_suffix": "page13.jpg"},
            {"text": "結局ヘクターはチケットを買い、ステージの前の方の席に座りました。「おや？ヒューゴが持っているのは、新しいバイオリンだ。私の古いバイオリンはどうしたんだろう」と思いましたが、そのとき演奏がスタートしました。「ああ、なんて素晴らしい！」それは、魂が揺さぶられるような、くるくる踊り出したくなるような、手を打ち鳴らしたくなるような、そんな演奏でした。「ヒューゴ！」思わずヘクターは叫びました。「私だよ。ヘクターだ。君は本当にすごい。誇りに思うよ」でもヒューゴは、ヒソヒソと何かをブラウンにささやいただけでした。ブラウンが少しだれかと話したようですが、演奏はそのまま続きました。", "image_path_suffix": "page14.jpg"},
            {"text": "数分後、ヘクターは、頑丈な腕に捉えられました。「一体、どう言うことだ？」怯えるヘクターを、ガードマンが暗い廊下の方へひきずり出します。「わかった。もう、ちょうど帰ろうとしていたところだ。話してくれ！」けれどガードマンは、両脇からヘクターを挟み、歩き続けます。そして、突然立ち止まりました。ヘクターは今自分がどこにいるのかわかりました。", "image_path_suffix": "page15.jpg"},
            {"text": "「さてさて、みなさん」と言うアナウンスの声が響きました。「今夜は、特別なゲストをお招きしております」「ヘクターさんに、どうぞ大きな拍手を！我らがスターのヒューゴは、ヘクターさんがいたからこそ誕生したのです」観客席が盛り上がる中、ヒューゴはヘクターに、古いバイオリンを手渡しました。この日のために、ずっと大切にしていたのです。ヒューゴはクーンと鳴いて、尻尾を振りました。バイオリンを受け取りながらヘクターは気付きました。ヒューゴと自分は、たとえ離れていても、その間もずっと友達だったのです。", "image_path_suffix": "page16.jpg"},
            {"text": "音楽は、いつも2人のそばに。そして2人の心も、いつもそばに。素晴らしい音楽が続くように、いつまでも2人は友達です。", "image_path_suffix": "page17.jpg"}
        ]
    }
}


# ==============================================================================
# 設定
# ==============================================================================
CURRENT_BOOK_ID = "inu"
JSON_FILE_PATH = f"story_{CURRENT_BOOK_ID}_emo.json"

# APIキーは環境変数で（安全のためここではプレースホルダ）
openai.api_key = os.getenv("OPENAI_API_KEY")

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    warnings.warn("環境変数 'GOOGLE_API_KEY' が未設定。テキスト感情分析は 0 扱いになります。", UserWarning)

client_options = ClientOptions(api_key=google_api_key or "")
client = language_v1.LanguageServiceClient(client_options=client_options)


# ==============================================================================
# 1. JSONファイルへの書き込み関数（既存仕様維持）
# ==============================================================================
def update_json_data(file_path: str, v_list: List[float], i_list: List[float], duration_list: List[float]):
    """
    既存のJSONリスト構造内の 'valence', 'intensity', 'flip_duration' を更新。
    先頭(1ページ目)はスキップ、末尾(最終ページ)は flip_duration=None。
    """
    try:
        if not os.path.exists(file_path):
            print(f"❌ エラー: JSONファイル '{file_path}' が見つかりませんでした。")
            return
        with open(file_path, "r", encoding="utf-8") as f:
            story_data = json.load(f)
        if not isinstance(story_data, list) or len(story_data) < 3:
            print(f"❌ エラー: '{file_path}' はリスト形式であるか、ページ数が少なすぎます。")
            return
    except json.JSONDecodeError:
        print(f"❌ エラー: '{file_path}' のJSON形式が不正です。")
        return
    except Exception as e:
        print(f"❌ JSONファイル読み込み中にエラー: {e}")
        return

    num_pages_total = len(story_data)
    num_analysis_pages = num_pages_total - 1  # 表紙を除く

    if not (len(v_list) == len(i_list) == len(duration_list) == num_analysis_pages):
        print(f"❌ エラー: 更新対象ページ数({num_analysis_pages}) と入力({len(v_list)}/{len(i_list)}/{len(duration_list)})が不一致。")
        print("💡 ヒント: JSONの総エントリ数-1（表紙を除いた数）と、BOOK_DEFINITIONSの'pages'の数が一致しているか確認してください。")
        return

    for i in range(1, num_pages_total):
        k = i - 1
        page_data = story_data[i]
        page_data["valence"] = round(v_list[k], 4)
        page_data["intensity"] = round(i_list[k], 4)

        if i == num_pages_total - 1:
            page_data["flip_duration"] = None
        else:
            page_data["flip_duration"] = int(round(duration_list[k] * 1000, 0))

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=4, ensure_ascii=False)
        print(f"🎉 ページ1と最終ページを除く全データを '{file_path}' に更新しました。")
    except Exception as e:
        print(f"❌ 書き込みエラー: {e}")


# ==============================================================================
# 2. Google Cloud Language API（テキスト感情）
# ==============================================================================
def analyze_text_sentiment(text_content: str):
    """テキストの (score, magnitude) を返す（未正規化の raw 値）。"""
    if not google_api_key:
        return 0.0, 0.0
    document = language_v1.Document(
        content=text_content,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language="ja",
    )
    encoding_type = language_v1.EncodingType.UTF8
    try:
        resp = client.analyze_sentiment(request={"document": document, "encoding_type": encoding_type})
        return resp.document_sentiment.score, resp.document_sentiment.magnitude
    except Exception as e:
        print(f"テキスト感情分析エラー: {e}")
        return 0.0, 0.0


# ==============================================================================
# 3. VQA（画像感情：BLIP-2）
# ==============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)
    blip_model.eval()
    vqa_ready = True
except Exception as e:
    print(f"BLIP-2モデルのロード失敗: {e}")
    vqa_ready = False

try:
    with open("positive_negative_question.json", "r", encoding="utf-8") as f:
        emotion_questions: Dict[str, List[Union[str, Dict[str, Union[str, float]]]]] = json.load(f)
except FileNotFoundError:
    warnings.warn("positive_negative_question.json が見つかりません。画像感情分析は 0 扱いになります。", UserWarning)
    emotion_questions = {}


@torch.no_grad()
def vqa_yes_probability(image: Image.Image, question: str) -> float:
    """BLIP-2 + FLAN-T5 で yes/no を判定し yes確率(0..1)を返す"""
    if not vqa_ready:
        return 0.0

    prompt = f"Question: {question}\nAnswer with 'yes' or 'no' only."
    inputs = blip_processor(image, prompt, return_tensors="pt").to(device)

    out = blip_model.generate(
        **inputs,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
    )

    logits = out.scores[0][0]
    tok = blip_processor.tokenizer
    yes_id = tok("yes", add_special_tokens=False).input_ids[0]
    no_id = tok("no", add_special_tokens=False).input_ids[0]

    two_logits = logits[[yes_id, no_id]]
    probs = torch.softmax(two_logits, dim=-1)
    return float(probs[0].item())


def normalize_questions(items: List[Union[str, Dict[str, Union[str, float]]]]) -> List[Dict[str, Union[str, float]]]:
    norm = []
    for it in items:
        if isinstance(it, str):
            norm.append({"question": it, "weight": 1.0})
        elif isinstance(it, dict):
            q = it.get("question")
            w = float(it.get("weight", 1.0))
            if q:
                norm.append({"question": q, "weight": w})
    return norm


def weighted_avg_score(image: Image.Image, items: List[Union[str, Dict[str, Union[str, float]]]]) -> float:
    qlist = normalize_questions(items)
    if not qlist:
        return 0.0
    wsum = 0.0
    wtot = 0.0
    for obj in qlist:
        q = obj["question"]  # type: ignore
        w = float(obj["weight"])  # type: ignore
        s = vqa_yes_probability(image, q)
        wsum += s * w
        wtot += w
    return wsum / wtot if wtot > 0 else 0.0


def smooth_scale(self_val: float, oppose_val: float,
                 s_min: float, s_max: float,
                 a_self: float, b_opp: float) -> float:
    t_self = max(0.0, min(1.0, self_val)) ** a_self
    t_opp = max(0.0, min(1.0, 1.0 - oppose_val)) ** b_opp
    t = max(0.0, min(1.0, t_self * t_opp))
    return s_min + (s_max - s_min) * t


def analyze_image_emotion(image_path: str):
    """画像から polarity(-1..+1) と intensity(0..1) を推定（raw）"""
    if not emotion_questions or not vqa_ready:
        return {"polarity": 0.0, "intensity": 0.0, "pos_raw": 0.0, "neg_raw": 0.0}

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"画像処理エラー({image_path}): {e}")
        return {"polarity": 0.0, "intensity": 0.0, "pos_raw": 0.0, "neg_raw": 0.0}

    POS_MIN, POS_MAX = 0.80, 1.20
    NEG_MIN, NEG_MAX = 0.80, 1.20
    A_POS, B_POS = 1.2, 1.6
    A_NEG, B_NEG = 1.2, 1.6
    MARGIN = 0.001
    EPS = 1e-6

    pos_raw = weighted_avg_score(image, emotion_questions.get("positive", []))
    neg_raw = weighted_avg_score(image, emotion_questions.get("negative", []))
    high = weighted_avg_score(image, emotion_questions.get("high_intensity", []))
    low  = weighted_avg_score(image, emotion_questions.get("low_intensity", []))

    pos_scale = smooth_scale(pos_raw, neg_raw, POS_MIN, POS_MAX, A_POS, B_POS)
    neg_scale = smooth_scale(neg_raw, pos_raw, NEG_MIN, NEG_MAX, A_NEG, B_NEG)

    pos = pos_raw * pos_scale
    neg = neg_raw * neg_scale

    d = pos - neg
    if abs(d) < MARGIN:
        polarity_raw = 0.0
    else:
        polarity_raw = (abs(d) - MARGIN) * (1 if d > 0 else -1) / (pos + neg + EPS)
        polarity_raw = max(-1.0, min(1.0, polarity_raw))

    intensity = (high + (1.0 - low)) / 2.0
    return {"polarity": polarity_raw, "intensity": intensity, "pos_raw": pos_raw, "neg_raw": neg_raw}


# ==============================================================================
# 4. OpenAI（ストーリー時間推定：in_page + gap）
# ==============================================================================
TIME_LABEL_TO_SECONDS: Dict[str, int] = {
    "一瞬": 5,
    "数秒": 15,
    "十数秒": 30,
    "数十秒": 45,
    "1分前後": 60,
    "数分": 180,
    "10分前後": 600,
    "数十分": 1800,
    "1時間前後": 3600,
    "数時間": 7200,
    "半日": 43200,
    "1日前後": 86400,
    "数日": 259200,
    "1週間前後": 604800,
    "数週間": 1209600,
    "数ヶ月": 5184000,
    "数年": 31536000,
}


def calculate_page_turn_time(story_seconds: int) -> float:
    """ストーリー経過秒数をページめくり時間 (0.60〜4.54秒) に写像。"""
    T_MIN, T_MAX = 0.60, 4.54
    L_MIN, L_MAX = 1.0, 8.0
    if not isinstance(story_seconds, (int, float)) or story_seconds <= 0:
        return T_MIN
    log_time = math.log10(story_seconds)
    N = (max(L_MIN, min(L_MAX, log_time)) - L_MIN) / (L_MAX - L_MIN)
    return round(T_MIN + N * (T_MAX - T_MIN), 2)


def _coerce_seconds(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip())
    except Exception:
        pass
    return default


def estimate_story_time_components(curr_text: str, next_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    現在ページの経過時間 in_page と、
    現在→次の間の経過時間 gap を別推定して合算する。
    """
    if not openai.api_key:
        return None

    allowed_labels = list(TIME_LABEL_TO_SECONDS.keys())
    allowed_str = ", ".join([f'"{k}"' for k in allowed_labels])

    next_text_safe = head(next_text, 100) if next_text else ""
    mapping_json = json.dumps(TIME_LABEL_TO_SECONDS, ensure_ascii=False, indent=2)

    prompt = f"""
あなたはプロの絵本読み聞かせボランティアです。
以下の「現在ページ」と「次ページ冒頭」を読み、ストーリー上の経過時間を厳格かつ一貫性をもって推定してください。

【出力する時間（2つ）】
1) in_page:
   現在ページの内容の中で進む時間。
   会話・行動・待ち時間・作業・感情のやり取りなど、
   そのページを読み進める間に物語世界で経過する時間。

2) gap:
   現在ページの最後から、次ページ冒頭までの“間”。
   明確な場面転換・時間跳躍・移動がある場合のみ発生する。

【重要な判断ルール】
- 次ページが以下のような表現で始まる場合、gap は長くしてはならない：
  - 「その日から」「翌朝」「次の日」「すぐに」「やがて（直後の意味）」
  → この場合、gap_duration は "なし" または "一瞬" を選ぶ。
- 「夕暮れ」「夜」「朝」「真夜中」「翌朝」などの時刻語が現在ページ内に含まれる場合、
  それは“時間が経過した”ことを強く示す。
  特に「夕暮れ→夜」「夜→翌朝」など時刻が跨ぐ描写は、
  in_page_duration を短く見積もってはならない（目安：夕暮れ→夜 は "1時間前後" 以上）。
- 「待つ」「心配で心配でならない」「帰ってこない」など“状態が続く”表現は、
  描写が短くても実時間が伸びるため in_page に時間を反映する。


【カテゴリ選択ルール】
- in_page_duration と gap_duration は必ず次から選ぶ: {allowed_str}
- 秒数は下の対応表どおりの整数にする
- 次ページが空（最終ページ相当）の場合は gap_duration="なし", gap_seconds=0

【カテゴリ→秒 対応表】
{mapping_json}

【現在ページ】
{curr_text}

【次ページ冒頭（参考）】
{next_text_safe}

【出力形式】
以下のJSONのみを出力すること（文章説明は禁止）：
{{
  "in_page_duration": "[カテゴリ]",
  "in_page_seconds": [整数],
  "gap_duration": "[カテゴリ or \\"なし\\"]",
  "gap_seconds": [整数],
  "reason": "なぜその in_page / gap と判断したかを20〜60字で説明"
}}
""".strip()

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content.strip())

        in_dur = data.get("in_page_duration", "不明")
        gap_dur = data.get("gap_duration", "不明")
        in_sec = _coerce_seconds(data.get("in_page_seconds"), default=0)
        gap_sec = _coerce_seconds(data.get("gap_seconds"), default=0)

        # ラベルから秒へ補正（モデルが秒をズラした時の安全策）
        if in_dur in TIME_LABEL_TO_SECONDS:
            in_sec = TIME_LABEL_TO_SECONDS[in_dur]
        if next_text is None:
            gap_dur = "なし"
            gap_sec = 0
        else:
            if gap_dur in TIME_LABEL_TO_SECONDS:
                gap_sec = TIME_LABEL_TO_SECONDS[gap_dur]

        reason = str(data.get("reason", "")).strip()

        return {
            "in_page_duration": in_dur,
            "in_page_seconds": int(in_sec),
            "gap_duration": gap_dur,
            "gap_seconds": int(gap_sec),
            "total_seconds": int(in_sec) + int(gap_sec),
            "reason": reason,
        }
    except Exception as e:
        print(f"時間推定エラー(components): {e}")
        return None


# ==============================================================================
# 4.5 感情によるめくり時間調整（あなたの元の関数）
# ==============================================================================
def recalculate_page_turning_time(
    T0: float,
    I: float,
    V: float,
    I_ref: float = 0.5,
    alpha_pos: float = 1.0,  # V>0 のときの短縮強度
    alpha_neg: float = 2.0,  # V<0 のときの延長強度
) -> float:
    """
    仕様:
    - V > 0:
        - I < I_ref なら T_final = T0
        - I > I_ref なら短くする (T_final < T0)
    - V < 0:
        - 常に長くする (T_final > T0)  ※Iに依存しない
    - V == 0:
        - 変化なし
    """
    T_MIN, T_MAX = 0.60, 4.54

    # 念のため入力を0..1 / -1..1にクリップ（上流で正規化済みでも安全）
    I = float(np.clip(I, 0.0, 1.0))
    V = float(np.clip(V, -1.0, 1.0))

    if abs(V) < 1e-12:
        return float(round(np.clip(T0, T_MIN, T_MAX), 2))

    # ----------------------------
    # V > 0: I_ref超過分だけ短縮
    # ----------------------------
    if V > 0:
        if I <= I_ref:
            T_raw = T0  # そのまま
        else:
            A1 = I - I_ref  # (0, 0.5] 想定
            k = 1.0 - alpha_pos * A1 * V
            T_raw = T0 * k

    # ----------------------------
    # V < 0: 常に延長（Iに依存しない）
    # ----------------------------
    else:  # V < 0
        # Vが負なので + にするため abs(V) を使う
        k = 1.0 + alpha_neg * abs(V)
        T_raw = T0 * k

    return float(round(np.clip(T_raw, T_MIN, T_MAX), 2))


# ==============================================================================
# 5. 実行
# ==============================================================================
if __name__ == "__main__":
    BOOK_CONFIG = BOOK_DEFINITIONS[CURRENT_BOOK_ID]
    book_pages_raw = BOOK_CONFIG["pages"]
    image_dir = BOOK_CONFIG["image_dir"]

    # book_pages リスト構築
    book_pages = []
    for page in book_pages_raw:
        full_image_path = os.path.join(image_dir, page["image_path_suffix"])
        book_pages.append({"text": page["text"], "image_path": full_image_path})

    # --- 未加工データ（raw）の収集リスト（名前だけ変更） ---
    raw_valence_text: List[float] = []
    raw_arousal_text: List[float] = []
    raw_valence_image: List[float] = []
    raw_arousal_image: List[float] = []
    base_turn_times: List[float] = []  # T0

    # JSON上のページ番号マップ（例：ページ3,5,7,...）
    num_content_pages = len(book_pages)
    json_page_num_map = [3 + 2 * i for i in range(num_content_pages)]

    print("--- 未加工データの収集 ---")
    for i, page in enumerate(book_pages):
        page_text = page["text"]
        image_path = page["image_path"]

        # ★ NEW: ページ内 + ページ間(gap) の時間を別で推定して合算
        next_text = book_pages[i + 1]["text"] if (i + 1 < len(book_pages)) else None
        comp = estimate_story_time_components(page_text, next_text)

        story_seconds = comp["total_seconds"] if comp else 10
        T0 = calculate_page_turn_time(story_seconds)
        base_turn_times.append(T0)

        # テキスト感情（raw）: 変数名だけ valence_text / arousal_text に変更
        valence_text, arousal_text = analyze_text_sentiment(page_text)
        raw_valence_text.append(valence_text)
        raw_arousal_text.append(arousal_text)

        # 画像感情（raw）: 変数名だけ valence_image / arousal_image に対応するリスト名へ変更
        img = analyze_image_emotion(image_path)
        raw_valence_image.append(img["polarity"])
        raw_arousal_image.append(img["intensity"])

        # ログ（名前だけ変更）
        time_log = ""
        if comp:
            time_log = (
                f"in={comp['in_page_duration']}({comp['in_page_seconds']}s) "
                f"+ gap={comp['gap_duration']}({comp['gap_seconds']}s) "
                f"= total={comp['total_seconds']}s"
            )
        else:
            time_log = "time_estimate=FAILED(default=10s)"

        print(
            f" P#{json_page_num_map[i]} | "
            f"{time_log} | "
            f"T0={T0:.2f} | "
            f"valence_text={valence_text:+.3f}, arousal_text={arousal_text:.3f} | "
            f"valence_image={img['polarity']:+.3f}, arousal_image={img['intensity']:.3f} | "
            f"pos_raw={img['pos_raw']:.3f}, neg_raw={img['neg_raw']:.3f}"
        )

        if comp and comp.get("reason"):
            print(f"    reason: {comp['reason']}")

    # バッチ正規化（名前だけ変更）
    valence_text_norm = normalize_signed(raw_valence_text)
    valence_image_norm = normalize_signed(raw_valence_image)

    arousal_text_norm01 = normalize_01(raw_arousal_text, fallback=0.0)
    arousal_image_norm01 = normalize_01(raw_arousal_image, fallback=0.0)

    # 統合 + 最終めくり
    final_valence_list: List[float] = []
    final_intensity_list_normalized: List[float] = []
    final_turning_time_list_adjusted: List[float] = []

    print("\n--- 正規化後の再計算 ---")
    for k in range(len(book_pages)):
        # 統合ロジックは変更なし（参照名だけ変更）
        V = (valence_text_norm[k] + valence_image_norm[k]) / 2.0
        A = max(arousal_text_norm01[k], arousal_image_norm01[k])
        T0 = base_turn_times[k]
        T_final = recalculate_page_turning_time(T0, A, V)

        final_valence_list.append(V)
        final_intensity_list_normalized.append(A)
        final_turning_time_list_adjusted.append(T_final)

        print(
            f" P#{json_page_num_map[k]} | "
            f"V={V:+.3f}, A={A:.3f}, T_final={T_final:.2f} "
            f"(arousal_text_raw={raw_arousal_text[k]:.3f}→norm={arousal_text_norm01[k]:.3f}, "
            f"arousal_image_raw={raw_arousal_image[k]:.3f}→norm={arousal_image_norm01[k]:.3f})"
        )

    # JSONへ書き込み
    if final_valence_list:
        print("\n--- JSONへ書き込み ---")
        update_json_data(
            file_path=JSON_FILE_PATH,
            v_list=final_valence_list,
            i_list=final_intensity_list_normalized,
            duration_list=final_turning_time_list_adjusted,
        )

    print("\n--- 統合分析プログラム終了 ---")
