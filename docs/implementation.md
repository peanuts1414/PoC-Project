# Implementation

## 1. 概要
　本ドキュメントは、製造データを対象とした異常検知 PoC の実装内容を整理したものです。  
　目的、リポジトリ構成、使用データ、実装方法、評価結果の概要を記載します。

 ● **目的**：製造データの異常検知をPoCとして試すことを目的にこのPoCを作成しました。\
　

◆**1.1 全体フロー**◆\
 \
<img width="1398" height="950" alt="Image" src="https://github.com/user-attachments/assets/0e07e116-24f9-4695-b5ba-fa0c902680d3" />
図１：全体フロー図
 

- 生データ取得：モータの<ins>電流値</ins>、<ins>速度</ins>、<ins>位置情報</ins>をPLCからCSV形式で取得
- データ前処理：ウィンドウ化や特徴量生成による整形
- モデル構築：<ins>Isolation Forest</ins>, <ins>One-Class-SVM</ins>, <ins>Autoencoder</ins>を使用
- 学習・推論：学習データでモデルを訓練し、評価データ（疑似異常を含む）で推論
- 評価：Accuracy, Precision, Recall, F1-scoreで性能を評価
  
 
◆**1.2 リポジトリ構成**◆\
　\
PoC-Project\
├─ README.md\
├─ Results/\
│　　├─ results_of_Autoencoder.png\
│　　├─ results_of_Isolation Forest.png\
│　　└─ results_of_One-Class-SVM.png\
├─ data/\
│　　├─ eval_features_scaled.csv　  # 評価用データ（前処理済み）\
│　　└─ train_features_scaled.csv　 # 学習用データ（前処理済み）\
├─ docs/\
│　　└─ implementation.md　        # 実装ドキュメント\
├─ notebooks/\
│　　└─ PoC_walkthrough.ipynb　  # 実装の Jupyter Notebook\
├─ src/\
│　　├─ Autoencoder\
│　　│　　├─ evaluate.py　     # Autoencoder 評価\
│　　│　　└─ train.py　        # Autoencoder 学習\
│　　├─ Isolation-Forest\
│　　│　　├─ evaluate.py　     # Isolation Forest 評価\
│　　│　　└─ train.py　        # Isolation Forest 学習\
│　　├─ One-Class-SVM\
│　　│　　├─ evaluate.py　     # One-Class-SVM 評価\
│　　│　　└─ train.py　        # One-Class-SVM 学習\
│　　├─ preprocessing\
│　　│　　├─ generate-JSON.py　      # JSONファイル作成\
│　　│　　├─ generate-features.py　  # 特徴量作成\
│　　│　　└─normalize-features.py　  # 特徴量正規化\
└─ requirements.txt         # 依存ライブラリ\
　




◆**1.3 評価結果**◆\
　\
![Image](https://github.com/user-attachments/assets/2b5234c7-93a9-49b5-a938-e25242dab737)\
表１：評価結果
<img width="1938" height="537" alt="Image" src="https://github.com/user-attachments/assets/7f991f7a-8d67-4782-9c37-e24e2315aeea" />
図２：評価結果グラフ\
 \
Accuracy：全体のうち、正しく分類できた割合\
Accuracy = (TP + TN) / (TP + TN + FP + FN)\
 \
Precision：異常と判定した中で、正しく異常だった割合\
Precision = TP / (TP + FP)\
 \
Recall：実際の異常のうち、正しく検出できた割合\
Recall = TP / (TP + FN)\
 \
F1-score：PrecisionとRecallのバランスをとった評価指標\
F1-score = 2 * (Precision * Recall) / (Precision + Recall)
  
- 評価結果の表を見るとどのモデルも高い精度を出せているとわかります。表からの評価だけでは選定しがたいです。
- グラフを確認すると、どれだけ分離できているかがわかります。特に、Autoencoderは正常と異常をはっきりと分離されていることが視認できます。


---

## 2. データ
 ◆**2.1 使用データ**◆

実際にモータを使っている機械を用いたデータを収集し、このPoCに用いました。

- データ量や制約条件
  - <ins>使ったデータは製品情報になりえるため社外への情報漏洩を考慮し、公開するデータは特徴量を正規化処理したあとのデータのみに限ります。</ins>
  - 異常データは実際に起こりえるケースを想定し、スケール、ノイズ、突発ピークの調整などを行い疑似異常を作成しました。

◆**2.2 データ使用範囲**◆\
　\
![Image](https://github.com/user-attachments/assets/8a23755b-7e39-48df-94a8-a46347de660b)
図３：使用範囲イメージ図

　上記の図３は、使用したデータ範囲を示すイメージ図になります。\
　今回使用したデータ範囲は、モータの起動時と停止時を除いた運転中のみになります。\
　停止と起動をデータに含んでしまうと、停止と起動が含まれたデータだけ外れ値になってしまい、上手く学習ができませんでした。\
　なので、停止と起動を含む場合は、停止と起動を含んだ一工程ごとの計測が必要になります。\
　もしくは、停止と起動のみを切り離して処理するなどの工夫が必要になります。


- 使用データの種類
  - モータ電流値：この値をベースに推論を行います。
  - モータ現在値：JSONファイル作成時に使います。
  - モータ速度：電流値と速度の相関関係を学習するのに用います。\
　\
![Image](https://github.com/user-attachments/assets/5c6dba7c-eaa8-4531-9ed7-654a15f319ad)
図４：モータ速度種類イメージ図

　上記の図４は、使用したモータ速度パターンを示すイメージ図になります。\
　パターンは2種類になります。\
　一つは速度一定、もう一つは速度を段階的に加速させたパターンになります。\
　ただし、速度自体はバリエーションを出すためにどれも速度の違うデータを用意しました。
    
◆**2.3 前処理方法（留意点のみ抜粋）**◆

- ウィンドウ幅について
![Image](https://github.com/user-attachments/assets/092b5fe9-032f-496f-86e7-531319b584b5)
図５：周期ズレイメージ図

　上記の図５は周期についてのイメージ図になります。周期一つにつきモータがワンサイクルしているイメージです。\
　今回の実装で一番工夫した点は、データをどのように特徴量に変換するかについてです。\
　特徴量を作成するのにあたり理想は、モータの挙動ワンサイクルにつき一つの特徴量を作成をすることです。\
　そうすることで正確に正常データと異常データを比較することができます。\
　そのためには、ウィンドウ幅を実際のモータの挙動に合わせた大きさに設定する必要があります。\
　失敗例として、ウィンドウ幅を固定にしてしまうと失敗する原因に繋がります。\
　理由として、モータ速度の変化やプロットのタイミングが合わないとモータの周期にズレが発生します。\
　周期がズレると特徴量の内容に大きく影響を及ぼし、正確にワンサイクルを比較することができなくなります。
 
**下記は羅列されたデータをモータ現在値によってワンサイクルごとにウィンドウ幅を区切るスクリプトです。**
- サイクル検出スクリプト
```
# ===== サイクル列からウィンドウ検出 =====
def detect_windows(cycle_values):
    start_rows = [0]
    for i in range(1, len(cycle_values)):
        if cycle_values[i] < cycle_values[i-1]:
            start_rows.append(i)
    windows = []
    for i in range(len(start_rows)):
        start = start_rows[i]
        end = start_rows[i+1] if i+1 < len(start_rows) else len(cycle_values)
        size = end - start
        windows.append((start, end, size))
    return windows

```
 上記のスクリプトにより、ワンサイクルの始まりを設定します。\
 今回使用したデータは0〜36000000をワンサイクルとし、このサイクルが繰り返されます。\
 ひとつ前より小さくなったタイミングがサイクルの始まりとみなし「start_row(サイクル始め)」を設定します。\
 次に、設定したワンサイクルの始まりと終わり、サイズを「windows」の中にタプルとして格納します。\
 サイズは単純に終わりから始まりを引いたものになります。

 - 下記出力例

   ```
   {
      CSV_win01
      "start_row": 30,
      "end_row": 91,
      "window_size": 61
   },
      CSV_win02
      "start_row": 91,
      "end_row": 153,
      "window_size": 62
   },
   ```
 
- ウィンドウフィルタリングスクリプト
```
def filter_windows(windows):
    if not windows:
        return []
    sizes = [w[2] for w in windows]
    avg_size = np.mean(sizes)
    min_size = avg_size * 0.8
    return [w for w in windows if w[2] >= min_size]
```
　上記のスクリプトはウィンドウサイズにフィルタリングをかけるスクリプトです。\
　データがワンサイクル分に達していないことを想定してこのスクリプトを設けました。\
　内容は、ウィンドウのサイズが平均の80％を下回ると切り捨てるようになっています。\
　例で言うと、0〜10までを繰り返すサイクルがあるとき、0～4までのデータが無いため5から始まってしまい、他のサイズより小さいものが出来上がってしまう場合です。\
　注意点として、データの中に大幅に速度の違うデータがあるとき、データ量に差ができてしまい必要なデータまでフィルタリングされてしまう場合があります。\
　今回は80％としきい値を設けましたが、この点に関しては速度ごとにしきい値を設けるなどの改善余地があるかと思います。

 - 特徴量抽出スクリプト

```
# ===== 特徴量抽出関数 =====
def extract_features(current_data, speed_data, window_size):
    feats = []
    for i in range(0, len(current_data), window_size):
        cur_win = current_data[i:i+window_size]
        spd_win = speed_data[i:i+window_size]
        if len(cur_win) == window_size and len(spd_win) == window_size:
            ratio = cur_win / np.where(spd_win == 0, 1, spd_win)  # ゼロ割防止
            feats.append({
                # 電流特徴量
                "current_mean": np.mean(cur_win),
                "current_std": np.std(cur_win),
                "current_min": np.min(cur_win),
                "current_max": np.max(cur_win),
                # 速度特徴量
                "speed_mean": np.mean(spd_win),
                "speed_std": np.std(spd_win),
                "speed_min": np.min(spd_win),
                "speed_max": np.max(spd_win),
                # 電流/速度比
                "cur_spd_ratio_mean": np.mean(ratio),
                "cur_spd_ratio_std": np.std(ratio),
                "cur_spd_ratio_min": np.min(ratio),
                "cur_spd_ratio_max": np.max(ratio),
            })
    return pd.DataFrame(feats)

```
上記は特徴量を抽出スクリプトです。\
特徴量にするのは、モータの電流と速度、それと電流と速度の比を使います。\
それらをウィンドウ幅ごとに<ins>平均値</ins>、<ins>分散</ins>、<ins>最小値</ins>、<ins>最大値</ins>を算出します。

 
---

## 3. 実装構成
- ディレクトリ構造（例: `src/`, `notebooks/`, `data/`）  
- 各ファイルの役割  

---

## 4. モデルと手法
- 使用モデル（Isolation Forest, One-Class SVM, Autoencoder）  
- モデルの概要説明  
- モデル選定の理由  

---

## 5. 評価指標
- 使用した指標（Accuracy, Precision, Recall, F1-score）  
- 指標選定の理由  

---

## 6. 実行方法
- 環境構築（`pip install -r requirements.txt`）  
- 実行コマンド（例: Jupyter Notebook の起動方法）  

---

## 7. 結果のまとめ
- 評価結果（スコアの概要）  
- 考察（得られた知見、制約、注意点）  

---

## 8. 今後の課題
- データ量拡大の必要性  
- 特徴量設計の改善  
- 実運用に向けた追加検証ポイント  

