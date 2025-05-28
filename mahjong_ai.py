# -*- coding: utf-8 -*-
"""
mahjong_ai_transformer.py  
==========================
TensorFlow 実装を **PyTorch + Deep Transformer** に総置換。

* **関数名は旧版 (make_model / train_ai / run_ai / test_ai / get_ai_dahai) のまま**
* 40,000 行の教師データでも過学習を避ける深層ネット (8 層 Transformer + Dropout + WeightDecay)
* `--resume` で途中再開可。各エポック終了時に完全チェックポイントを保存

データ形式
-----------
CSV / TXT 1 行 1 サンプル、フォーマットは
```
14枚牌文字列 打牌牌文字列
例) 1m1m1m2m3m6m7m5p6p7p8s9s9s9s 8s
```

依存
----
```
pip install torch torchmetrics tqdm
```

"""
import argparse
from pathlib import Path
from typing import List, Tuple, Any, Dict
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm

import mahjong_common as mjc
import mahjong_loader as mjl

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# ----------------- 定数 -----------------
NUM_HAI = 34          # 牌種
TOKEN_PAD = 34        # pad トークン id
MAX_HAND = 14         # 牌枚数 (常に 14)
CKPT_DIR = Path("ckpt")
CKPT_DIR.mkdir(exist_ok=True)

# ----------------- 牌＜＞ID 変換 -----------------
_suit_map = {'m': 0, 'p': 9, 's': 18, 'z': 27}
_id_map_cache: Dict[str, int] = {}

# 漢字牌の置換テーブル
zhai_map = {"東": "1z", "南": "2z", "西": "3z", "北": "4z", "白": "5z", "發": "6z", "発": "6z", "中": "7z"}

def hai_str_to_id(hai: str) -> int:
    if hai in _id_map_cache:
        return _id_map_cache[hai]
    rank, suit = hai[0], hai[1]
    base = _suit_map[suit]
    _id_map_cache[hai] = base + int(rank) - 1
    return _id_map_cache[hai]

import random  # ← 必要ならクラス外で追加しておいてOK

class MahjongDataset(Dataset):
    def __init__(self, path: Path):
        self.samples: List[Tuple[List[int], int]] = []
        with path.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                for k, v in zhai_map.items():
                    ln = ln.replace(k, v)
                try:
                    hand_str, out_str = ln.split()
                    if len(hand_str) != 28 or len(out_str) != 2:
                        continue
                    tiles = [hand_str[i:i + 2] for i in range(0, 28, 2)]
                    random.shuffle(tiles)  # ← ⭐ここ！

                    if any(len(t) != 2 for t in tiles):
                        continue
                    ids = [hai_str_to_id(t) for t in tiles]
                    if len(ids) != 14:
                        continue
                    dahai = hai_str_to_id(out_str)
                    self.samples.append((ids, dahai))
                except Exception as e:
                    print(f"⚠️ 行処理中に例外: {ln} → {e}")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ----------------- モデル -----------------
class MahjongTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_head: int = 8, num_layers: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.15):
        super().__init__()
        self.embed = nn.Embedding(NUM_HAI + 1, d_model)  # + PAD
        self.pos_emb = nn.Parameter(torch.randn(MAX_HAND, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NUM_HAI)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,14) long
        mask = x == TOKEN_PAD
        h = self.embed(x) + self.pos_emb  # (B,14,D)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.norm(h)
        h = h.mean(dim=1)  # (B,D)
        return self.head(h)  # (B,34)

# ----------------- グローバル変数 -----------------
model: MahjongTransformer | None = None
optimizer: AdamW | None = None
scheduler: CosineAnnealingLR | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acc_metric = MulticlassAccuracy(num_classes=NUM_HAI, average="micro").to(device)

# ----------------- 旧関数名 API -----------------

def make_model() -> None:
    """モデル・optimizer・scheduler を初期化しグローバルに設定"""
    global model, optimizer, scheduler
    model = MahjongTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)  # 仮。train_ai() 内で set

# --------- 内部ユーティリティ ---------

def _save_ckpt(epoch: int, path: Path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
    }, path)


def _load_ckpt(path: Path) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    scheduler.load_state_dict(ckpt["sched"])
    return ckpt["epoch"]

# --------- 学習 ---------

def train_ai(filename: str, num_epochs: int):
    path = Path(filename)
    assert path.exists(), f"データファイルが見つかりません: {filename}"

    full_ds = MahjongDataset(path)
    val_size = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    scheduler.T_max = 1000  # 適当に大きくしておく（周期制御）

    criterion = nn.CrossEntropyLoss()
    ckpt_path = CKPT_DIR / "mahjong_transformer.pt"

    start_epoch = 0
    if ckpt_path.exists():
        start_epoch = _load_ckpt(ckpt_path)
        print(f"[Resume] {start_epoch} エポック目から再開…")

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):  # ← 修正！
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} - train", leave=True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        acc_metric.reset()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                acc_metric.update(preds, yb)
        val_acc = acc_metric.compute().item()
        print(f"Epoch {epoch:3d}: loss {train_loss:.4f}, val acc {val_acc:.4%}")

        _save_ckpt(epoch, ckpt_path)

    print(f"[完了] 学習終了。追加で {num_epochs} エポック学習しました")


from tqdm import tqdm

def reinforce_train(num_games: int = 100, lambda_pg: float = 1.0):
    """
    run_ai() のようにプレイしながら、REINFORCE によって fine-tuning。
    `lambda_pg` は policy gradient loss の重み。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    log_probs = []
    rewards = []
    win_count = 0
    tenpai_count = 0

    for i in tqdm(range(num_games), desc="[強化学習] Simulating"):
        mjc.init_yama()
        tehai = mjc.get_haipai()
        tsumo_count = 0
        total_reward = 0.0

        tenpai = False  # ← while の前にこれを追加

        # ループ中に一度でもテンパイしたかを記録
        while tsumo_count < 18:
            tsumo = mjc.get_tsumo()
            if tsumo == -1:
                break
            tsumo_count += 1
            tehai[tsumo] += 1

            if mjc.is_agari(tehai):
                total_reward += 200.0
                win_count += 1
                break

            if not tenpai and mjc.is_tenpai(tehai):
                tenpai = True

            ids = _tehai_vector_to_idlist(tehai)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)
            probs = F.softmax(logits[0], dim=0)
            probs_masked = probs.clone()
            for i in range(NUM_HAI):
                if tehai[i] == 0:
                    probs_masked[i] = 0.0
            probs_masked /= probs_masked.sum() + 1e-6

            dist = Categorical(probs_masked)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            tehai[action.item()] -= 1

        # ======== 報酬計算（ループ後）========
        if total_reward == 0:
            if mjc.is_tenpai(tehai):
                total_reward = 200.0
                tenpai_count += 1
            elif tenpai:
                total_reward = -5000.0  # テンパイしてたのに崩れた
            else:
                total_reward = -100.0   # 一度もテンパイできず終局

        rewards.append(total_reward)

    # normalize rewards
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    policy_loss = 0.0
    for log_prob, reward in zip(log_probs, rewards):
        policy_loss += -log_prob * reward
    policy_loss /= len(rewards)

    # 🔥 entropy bonus を加える（学習の偏り抑制）
    entropy = -torch.sum(probs_masked * torch.log(probs_masked + 1e-6))
    policy_loss -= 0.01 * entropy  # ← ここが追加点！

    # cross-entropy loss（1バッチだけでOK）
    path = Path("dahai_data.txt")
    full_ds = MahjongDataset(path)
    loader = DataLoader(full_ds, batch_size=512, shuffle=True, num_workers=2)
    ce_loss_total = 0.0
    criterion = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            logits = model(xb)
            ce_loss = criterion(logits, yb)
            ce_loss_total += ce_loss.item()
        break

    total_loss = ce_loss_total + lambda_pg * policy_loss
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # 結果表示
    print(f"[強化学習] 和了: {win_count} / {num_games}, 聴牌: {tenpai_count}")
    print(f"[強化学習] CE loss: {ce_loss_total:.4f}, PG loss: {policy_loss.item():.4f}, total: {total_loss.item():.4f}")



# --------- 推論 ---------

def _tehai_vector_to_idlist(tehai: List[int]) -> List[int]:
    """34 長の枚数ベクトル → 14 個の id list"""
    ids = []
    for idx, cnt in enumerate(tehai):
        ids.extend([idx] * cnt)
    assert len(ids) == 14, "手牌枚数が 14 枚ではありません"
    return ids


def get_ai_dahai(ai_in: List[int], ai_out: torch.Tensor) -> int:
    """ai_out (34 logits) に基づき、ai_in に存在する牌で最大確率のものを返す"""
    scores = ai_out.clone()
    while True:
        idx = torch.argmax(scores).item()
        if ai_in[idx] > 0:
            return idx  # 打牌決定
        scores[idx] = -math.inf  # その牌が無いので次点へ


def test_ai():
    """1 局シミュレーションして結果を返す"""
    mjc.init_yama()
    tehai = mjc.get_haipai()
    tsumo_count = 0
    tenpai = False
    
    while tsumo_count < 18:
        tsumo = mjc.get_tsumo()
        if tsumo == -1:
            break
        tsumo_count += 1
        tehai[tsumo] += 1
        if mjc.is_agari(tehai):
            tehai[tsumo] -= 1
            info = mjc.AgariInfo(tehai, tsumo)
            return info.get_point(), info.get_yaku_strings(), False

        ids = _tehai_vector_to_idlist(tehai)
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))
        dahai = get_ai_dahai(tehai, logits[0])
        tehai[dahai] -= 1
        if not tenpai and mjc.is_tenpai(tehai):
            tenpai = True

    if mjc.is_tenpai(tehai):
        return 0, [], False
    return -1, [], tenpai


def train_loop_alternating(
    datafile: str = "dahai_data.txt",
    alt_count: int = 10,
    reinforce_games: int = 100,
    epochs_each: int = 0,
):
    """
    強化学習と教師あり学習を交互に繰り返すループ
    datafile       : 教師ありデータファイル
    alt_count      : 強化学習 ⇄ 教師あり学習 の交互回数
    reinforce_games: 1回あたりの強化学習の局数
    epochs_each    : 1回あたりの教師あり学習エポック数
    """
    make_model()
    ckpt = CKPT_DIR / "mahjong_transformer.pt"
    if ckpt.exists():
        _load_ckpt(ckpt)

    for i in range(1, alt_count + 1):
        print(f"\n🧠 === Step {i}: 強化学習 {reinforce_games}局 ===")
        reinforce_train(num_games=reinforce_games)

        print(f"\n📚 === Step {i}: 教師あり学習 {epochs_each}エポック ===")
        train_ai(datafile, num_epochs=epochs_each)

    print("\n✅ 全スケジュール完了")


# 最後の __main__ 部分のみ完成版として記載
# 上部は既存コードそのままでOK（reinforce_loop_multiple の追加も含む）

import time  # 🔽 休憩用 sleep を使うために必要

# 🔽 強化学習を複数回繰り返す関数

def reinforce_loop_multiple(times: int = 5, games_each: int = 100, sleep_sec: int = 30, lambda_pg: float = 1.0):
    """
    強化学習を times 回繰り返し、各回ごとに sleep_sec 秒休憩する。
    lambda_pg は policy gradient loss の重み。
    """
    make_model()
    ckpt = CKPT_DIR / "mahjong_transformer.pt"
    if ckpt.exists():
        _load_ckpt(ckpt)

    for i in range(times):
        print(f"\n\U0001f9e0 === 強化学習 {i+1}/{times} 回目 ({games_each} 局, \u03bb={lambda_pg}) ===")
        reinforce_train(num_games=games_each, lambda_pg=lambda_pg)
        if i < times - 1:
            print(f"\u23f8 次の強化学習まで {sleep_sec} 秒休憩...")
            time.sleep(sleep_sec)

    print(f"\n\u2705 全 {times} 回の強化学習完了！")


def run_ai():
    """AI を 10,000 局テストし統計出力"""
    test_count = 100
    agari = tenpai_ru = tenpai_kuzu = total_pts = 0
    yaku_count: Dict[str, int] = {}

    for _ in tqdm(range(test_count), desc="Sim", leave=False):
        pts, yakus, tenpai_kuzusi = test_ai()
        if pts > 0:
            agari += 1
            total_pts += pts
            for y in yakus:
                yaku_count[y] = yaku_count.get(y, 0) + 1
        elif pts == 0:
            tenpai_ru += 1
        elif tenpai_kuzusi:
            tenpai_kuzu += 1

    print("和了:", agari)
    if agari:
        print("平均得点:", total_pts / agari)
        print("役内訳:", yaku_count)
    print("流局時聴牌:", tenpai_ru)
    print("聴牌崩し:", tenpai_kuzu)

# mahjong_common.py に追記
def hai_str_to_id(hai: str) -> int:
    suit_map = {'m': 0, 'p': 9, 's': 18, 'z': 27}
    zhai_map = {"東": "1z", "南": "2z", "西": "3z", "北": "4z", "白": "5z", "發": "6z", "発": "6z", "中": "7z"}
    hai = zhai_map.get(hai, hai)
    return suit_map[hai[1]] + int(hai[0]) - 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", action="store_true", help="AI シミュレーションを実行")
    parser.add_argument("-t", "--train", action="store_true", help="AI を学習")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="学習エポック数")
    parser.add_argument("-d", "--datafile", default="dahai_data.txt", help="学習データファイル")

    parser.add_argument("--alternate", action="store_true", help="教師ありと強化学習を交互に実行 (2回)")
    parser.add_argument("--alternate20", action="store_true", help="強化学習100局 + 教師あり10ep を20回交互に実行")

    parser.add_argument("--reinforce-loop", action="store_true", help="強化学習を複数回連続で実行")
    parser.add_argument("--loop-times", type=int, default=10, help="強化学習の繰り返し回数")
    parser.add_argument("--loop-sleep", type=int, default=5, help="各強化学習後の休憩秒数")
    parser.add_argument("--lambda-pg", type=float, default=1.0, help="Policy Gradient loss の重み λ")

    args = parser.parse_args()

    # モデル初期化・チェックポイント読み込み
    make_model()
    ckpt = CKPT_DIR / "mahjong_transformer.pt"
    if ckpt.exists():
        _load_ckpt(ckpt)
        print("[✅] モデルをチェックポイントから読み込みました")
    else:
        print("[⚠] チェックポイントが存在しません（ランダム初期化）")

    # 通常の教師あり学習
    if args.train:
        train_ai(args.datafile, args.epochs)

    # 推論テスト（10,000局）
    if args.run:
        run_ai()

    # 教師ありと強化学習の交互トレーニング（2回だけ）
    if args.alternate:
        train_loop_alternating(
            datafile=args.datafile,
            alt_count=2,
            reinforce_games=100,
            epochs_each=1
        )

    # 強化学習100局 + 教師あり10ep を20回交互に実行（本命）
    if args.alternate20:
        train_loop_alternating(
            datafile=args.datafile,
            alt_count=100,
            reinforce_games=100,
            epochs_each=2
        )

    # 強化学習だけを連続実行
    if args.reinforce_loop:
        reinforce_loop_multiple(
            times=args.loop_times,
            games_each=100,
            sleep_sec=args.loop_sleep,
            lambda_pg=args.lambda_pg
        )


