# **Swimming Squid Battle** 魷來魷去對戰版

![swimming-squid-battle](https://img.shields.io/github/v/tag/PAIA-Playful-AI-Arena/swimming-squid-battle)

[![MLGame](https://img.shields.io/badge/MLGame->10.6.a9-<COLOR>.svg)](https://github.com/PAIA-Playful-AI-Arena/MLGame)

[//]: # (![Logo]&#40;https://raw.githubusercontent.com/PAIA-Playful-AI-Arena/swimming_squid_battle/refs/tags/1.6.1/asset/logo.png&#41;)
這是一個魷魚吃東西小遊戲，茫茫的海洋中有美味的食物，也有人類拋棄的垃圾，還出現了搶食物的同類！！
    請用你的AI幫助小小魷魚，面對雜亂的海洋，面對同類的競爭快快長大。

![demo](https://raw.githubusercontent.com/PAIA-Playful-AI-Arena/swimming_squid_battle/refs/tags/1.6.2/asset/banner.gif)

---

# 基礎介紹

## 啟動方式

- 直接啟動 [main.py](https://github.com/PAIA-Playful-AI-Arena/swimming_squid_battle/blob/1.6.2/main.py) 即可執行

## 遊戲參數設定

```python
# main.py 
game = SwimmingSquidBattle(
            level: int = 1,
            level_file: str = None,
            game_times: int = 1)
```

- `level`: 選定內建關卡，預設為 1 選擇第一關。
- `level_file`: 使用外部檔案作為關卡，請注意，使用此設定將會覆蓋掉關卡編號，並且不會自動進入下一關。
- `game_times`：選擇要對戰幾次決勝負，可選擇一戰決勝負、三戰兩勝制、五戰三勝制。預設為一戰決勝負。


## 玩法

- 1P：使用鍵盤 上、下、左、右 控制魷魚
- 2P：使用鍵盤 W、S、A、D 控制魷魚

## 遊戲規則

### 角色升級機制

角色初始等級皆為 1，隨著得分增加升 / 降級。等級將會影響角色長寬與移動速度，各等級對應資料如下：

| Lv| 分數 | 角色寬度 | 角色高度 |移動速度|
| --- |------| ------- | ------ | ------|
| 1 | ~9   | 30      | 60     |25|
| 2 | 10~29   | 36      | 72     |21|
| 3 | 30~59   | 42      | 84     |18|
| 4 | 60~99  | 48      | 96     |16|
| 5 | 100~149  | 54      | 108     |12|
| 6 | 150~  | 60      | 120     |9|

### 得分 / 扣分規則

1. 吃東西：
   1. 角色可以透過吃海裡漂浮的東西獲取分數，但海裡也有垃圾存在，吃到垃圾將會扣分。
   2. 不同的食物 / 垃圾會有不同的大小與分數。資料如下：

      |食物名稱 | 物件寬度 | 對應分數 | 移動速度 |
      | ----- | ------- | ------ | ------ |
      | FOOD_1 | 30      | 1     | 1      |
      | FOOD_2 | 40      | 2     | 2      |
      | FOOD_3 | 50      | 4     | 4      |
      | GARBAGE_1 | 30      | -1     | 1      |
      | GARBAGE_3 | 40      | -4     | 2      |
      | GARBAGE_3 | 50      | -10     | 4      |

   3. 食物數量會隨遊戲時間增加，直到上限。
   4. 垃圾由上往下掉落，並且會漂浮晃動。
   5. 食物會在畫面隨機出現，並且會隨機移動，且不會直接出現在魷魚身上。

2. 玩家相撞：
   1. 兩隻魷魚相撞時，如果一方等級較高，則等級高者加 `6` 分，等級低者扣 `6` 分。
   2. 如果兩方等級相同，則雙方皆扣 `5` 分。
   3. 受傷那一方，會麻痺 `8` frame，過程中隨機向一個方向移動，移動期間，不會受到攻擊，也不會吃任何食物。
   4. 受傷那一方，麻痺效果解除後，將會有 `30` frame 的無敵時間，不會再受到對手的攻擊，但可以吃食物。

### 獲勝條件

1. 時間結束前，先達到`目標分數`者獲勝。
2. 時間結束前，兩方分數相同，將會延長遊戲時間 `300` frame、增加垃圾數量。
3. 若兩人同時通關，分數較高者勝。
4. 若兩人同時通關分數相同，將會延長遊戲時間 `300` frame、增加垃圾數量，提高 `目標分數` `50` 分。
5. 遊戲沒有平手，依照上述規則判定，直到分出勝負。

---

# 座標系統

1. 使用 pygame 座標系，`左上角`為原點，`X軸`往`右`為正，`Y軸`往`下`為正
2. 回傳的物件座標，皆為物體`中心點`座標

# 進階說明

## 使用ＡＩ玩遊戲

```bash
# 在easy game中，打開終端機
python -m mlgame -i ./ml/ml_play_template.py ./ --level 3
python -m mlgame -i ./ml/ml_play_template.py ./ --level_file /path_to_file/level_file.json
```

## ＡＩ範例

```python
import random

class MLPlay:
    def __init__(self,ai_name,*args, **kwargs):
        print("Initial ml script")

    def update(self, scene_info: dict,,*args, **kwargs):

        # print("AI received data from game :", scene_info)

        actions = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]

        return random.sample(actions, 1)

    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass
```

## 遊戲資訊

- scene_info 的資料格式如下

```json
{
  "frame": 15,
  "score": 8,
  "score_to_pass": 10,
  "self_x": 100,
  "self_y": 300,
  "self_w": 30,
  "self_h": 45,
  "self_vel": 25,
  "self_lv": 1,
  "opponent_x":500,
  "opponent_y":400,
  "opponent_lv": 2,
  "status": "GAME_ALIVE",
  "foods": [
    {
      "h": 30,
      "score": 1,
      "type": "FOOD_1",
      "w": 30,
      "x": 40,
      "y": 134
    },
    {
      "h": 40,
      "score": 2,
      "type": "FOOD_2",
      "w": 40,
      "x": 422,
      "y": 192
    },
    {
      "h": 50,
      "score": 4,
      "type": "FOOD_3",
      "w": 50,
      "x": 264,
      "y": 476
    },
    {
      "h": 30,
      "score": -1,
      "type": "GARBAGE_1",
      "w": 30,
      "x": 100,
      "y": 496
    },
    {
      "h": 40,
      "score": -4,
      "type": "GARBAGE_2",
      "w": 40,
      "x": 633,
      "y": 432
    },
    {
      "h": 50,
      "score": -10,
      "type": "GARBAGE_3",
      "w": 50,
      "x": 54,
      "y": 194
    }
  ] ,
  "env": {
    "time_to_play": 600,
    "playground_size_w":700, 
    "playground_size_h":550
  }

}
```

- `frame`：遊戲畫面更新的編號。
- `self_x`：玩家角色的Ｘ座標，表示方塊的`中心點`座標值，單位 pixel。
- `self_y`：玩家角色的Ｙ座標，表示方塊的`中心點`座標值，單位 pixel。
- `self_w`：玩家角色的寬度，單位 pixel。
- `self_h`：玩家角色的高度，單位 pixel。
- `self_vel`：玩家角色的速度，表示方塊每幀移動的像素，單位 pixel。
- `self_lv`：玩家角色的等級，最小 1 ，最大 6。
- `opponent_x`：對手角色的Ｘ座標，表示方塊的`中心點`座標值，單位 pixel。
- `opponent_y`：對手角色的Ｙ座標，表示方塊的`中心點`座標值，單位 pixel。
- `opponent_lv`：對手角色的等級，最小 1 ，最大 6。
- `foods`：食物的清單，清單內每一個物件都是一個食物的`中心點`座標值，也會提供此食物是什麼類型和分數多少。
  - `type` 食物類型： `FOOD_1`, `FOOD_2`, `FOOD_3`, `GARBAGE_1`, `GARBAGE_2`, `GARBAGE_3`
- `score`：目前得到的分數
- `score_to_pass`：通關分數
- `env`：環境資訊，裡面會包含遊戲設定檔的所有參數，若有不符範圍的，會拿到校正後的參數。
- `status`： 目前遊戲的狀態
  - `GAME_ALIVE`：遊戲進行中
  - `GAME_PASS`：遊戲通關
  - `GAME_OVER`：遊戲結束

## 動作指令

- 在 update() 最後要回傳一個字串，主角物件即會依照對應的字串行動，一次只能執行一個行動。
  - `UP`：向上移動
  - `DOWN`：向下移動
  - `LEFT`：向左移動
  - `RIGHT`：向右移動
  - `NONE`：原地不動

## 遊戲結果

- 最後結果會顯示在console介面中，若是PAIA伺服器上執行，會回傳下列資訊到平台上。

```json
{
  "frame_used": 100,
  "status": "fail",
  "attachment": [
    {
      "squid": "1P",
      "score": 0,
      "rank": 2,
      "wins": "1 / 3"
    },
    {
      "squid": "2P",
      "score": 10,
      "rank": 1,
      "wins": "2 / 3"
    }
  ]
}
```

- `frame_used`：表示使用了多少個frame
- `status`：表示遊戲結束的狀態
  - `fail`:遊戲過程出現問題
  - `finish`:遊戲結束，回傳完成
- `attachment`：紀錄遊戲各個玩家的結果與分數等資訊
  - `squid`：玩家編號
  - `score`：吃到的食物總數
  - `rank`：排名
  - `wins`：目前勝場數

## 自定義環境地圖

[地圖設定檔](https://github.com/PAIA-Playful-AI-Arena/swimming_squid_battle/blob/main/levels/template.json)
可修改檔案中的參數，並匯入遊戲中使用。

```json
{
  "time_to_play": 600, //遊戲時間限制
  "playground_size_w":1200, //環境寬度，需要介於 100~1200
  "playground_size_h":650, //環境寬度，需要介於 100~650
  "score_to_pass": 10, //通關分數
  "food_1": 3, //初始食物數量 
  "food_1_max": 5, //最大食物數量
  "food_2": 2,
  "food_2_max": 3,
  "food_3": 5,
  "food_3_max": 7,
  "garbage_1": 1,
  "garbage_1_max": 4,
  "garbage_2": 1,
  "garbage_2_max": 3,
  "garbage_3": 1,
  "garbage_3_max": 2

}
```

---

# 參考資源

- 音效與音樂
  - https://amachamusic.chagasi.com/index.html
  - https://soundeffect-lab.info/
  - http://www.kurage-kosho.info/
  - https://youfulca.com/
  - https://mart.kitunebi.com/
  - https://taira-komori.jpn.org/
