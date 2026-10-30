# EEG Classification with EEGNet and DeepConvNet

本專案使用 PyTorch 實作 EEG 二分類，提供 EEGNet 與 DeepConvNet，並支援訓練/驗證、一次性 test evaluation、訓練紀錄與曲線輸出。模型與訓練入口位於 `models/EEGNet.py`、`main.py`；資料載入與前處理位於 `dataloader.py`。

## 功能簡介

- EEGNet：可調整 activation、ELU alpha、dropout、`f1`、depth multiplier `d`、`f2`、時間卷積 kernel 與兩次 average pooling 寬度。
- DeepConvNet：依本專案表格實作的四段卷積架構，支援 activation、ELU alpha、dropout。
- 從 training set 依類別分層切出 validation set；比例由 `--validation-split` 設定。
- 每個 epoch 記錄 training accuracy/loss 與 validation accuracy/loss。
- 訓練結束後載入選出的 checkpoint，對 test set 評估一次。
- 每次執行結果另存於 `result/<run_name>/`，不必覆蓋其他實驗。

## 環境需求

- Python 3.9 或更新版本
- PyTorch（有可用 CUDA 時會自動使用 GPU，否則使用 CPU）
- NumPy
- Matplotlib
- tqdm
- torchsummary（目前 `main.py` 有匯入此套件）

安裝必要套件範例：

```bash
python -m pip install torch numpy matplotlib tqdm torchsummary
```

依作業系統與 CUDA 版本選擇合適的 PyTorch 安裝方式。專案目前沒有 `requirements.txt`，執行前請確認以上套件已安裝。

## 資料格式與位置

`dataloader.py` 會以相對路徑讀取以下四個 NumPy `.npz` 檔案：

```text
lab2_EEG_classification/
└── data/
    ├── S4b_train.npz
    ├── S4b_test.npz
    ├── X11b_train.npz
    └── X11b_test.npz
```

請從包含 `main.py` 和 `lab2_EEG_classification/data/` 的專案根目錄執行命令，讓上述相對路徑正確解析。每個 `.npz` 檔案需包含 `signal` 與 `label` 欄位；載入器會合併 S4b 與 X11b 的 train/test data，將 label 從 1-based 轉為 0-based，並將訊號轉為 `(trial, 1, channel, sample)`。NaN 會以對應 train 或 test 資料中的全域 `nanmean` 補值。

本專案目前使用 2 個 EEG channels、每 trial 750 samples、2 個分類類別。資料集檔案需由使用者依課程/競賽提供方式取得；README 不包含原始資料。

## 執行方式

從專案根目錄執行。以下是使用 EEGNet 的範例：

```bash
python main.py \
  -model eegnet \
  -num_epochs 200 \
  -batch_size 64 \
  -lr 0.001 \
  -optimizer adam \
  -activation leaky_relu \
  -elu_alpha 1.0 \
  -dropout 0.25 \
  -validation_split 0.05 \
  -seed 42 \
  -f1 16 -d 3 -f2 32 \
  --temporal-kernel 51 \
  --separable-kernel 15 \
  --pool1 8 --pool2 8 \
  -run_name eegnet_example
```

Windows PowerShell 可寫成單行：

```powershell
python main.py -model eegnet -num_epochs 200 -batch_size 64 -lr 0.001 -optimizer adam -activation leaky_relu -elu_alpha 1.0 -dropout 0.25 -validation_split 0.05 -seed 42 -f1 16 -d 3 -f2 32 --temporal-kernel 51 --separable-kernel 15 --pool1 8 --pool2 8 -run_name eegnet_example
```

DeepConvNet 範例：

```bash
python main.py -model deepconvnet -num_epochs 150 -batch_size 64 -lr 0.01 -optimizer adam -activation elu -elu_alpha 1.0 -dropout 0.5 -validation_split 0.1 -seed 42 -run_name deepconvnet_example
```

### 命令列參數

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `-model`, `--model` | `eegnet` | 模型：`eegnet` 或 `deepconvnet` |
| `-num_epochs` | 150 | 訓練 epoch 數，需大於 0 |
| `-batch_size` | 64 | Batch size，需大於 0 |
| `-lr` | 0.01 | Optimizer learning rate |
| `-optimizer`, `--optimizer` | `adam` | `adam` 或 `sgd`；SGD 使用 momentum=0.9 |
| `-activation`, `--activation` | `relu` | `elu`、`relu` 或 `leaky_relu` |
| `-elu_alpha`, `--elu-alpha` | 1.0 | ELU alpha；其他 activation 不使用此值 |
| `-dropout`, `--dropout` | 0.25 | Dropout probability |
| `-validation_split`, `--validation-split` | 0.1 | 從原 training set 切出的 validation 比例，必須介於 0 與 1 |
| `-seed`, `--seed` | 42 | 用於 validation split、DataLoader shuffle 與 PyTorch seed |
| `-run_name`, `--run-name` | 自動產生 | `result/` 下的輸出資料夾名稱 |
| `-f1`, `--f1` | 16 | EEGNet 第一層時間 filter 數 |
| `-d`, `--depth-multiplier` | 2 | EEGNet depth multiplier |
| `-f2`, `--f2` | `None` | EEGNet separable 卷積輸出 filters；未指定時為 `f1*d` |
| `--temporal-kernel` | 51 | EEGNet 第一層時間卷積 kernel width |
| `--separable-kernel` | 15 | EEGNet depthwise separable 時間卷積 kernel width |
| `--pool1` | 4 | EEGNet 第一次 average pooling 寬度與 stride |
| `--pool2` | 8 | EEGNet 第二次 average pooling 寬度與 stride |

所有 kernel/pooling/filter 寬度需為正整數。`f1`、`d`、`f2` 是 EEGNet 專用參數；選 DeepConvNet 時不會改變該網路結構。Optimizer 的 weight decay 固定為 0.001，loss 固定使用 `torch.nn.CrossEntropyLoss()`。

## Best checkpoint 選擇方式

每個 epoch 會計算 validation accuracy 與 validation cross-entropy loss。程式會在 **validation accuracy 不低於目前最佳值**，且 **validation loss 低於目前最佳值** 時保存 checkpoint。因此它不是單純挑最高 validation accuracy，也不是單純挑最低 validation loss；平手 accuracy 時仍可由更低 loss 更新 checkpoint。訓練後會載入該 checkpoint 再計算 test accuracy。

同一個 seed 會控制 stratified validation split 與 DataLoader shuffle/PyTorch random seed。不同 seed 的結果因此同時受到 validation subset、資料順序及模型隨機性的影響。Test set 每次 run 都會被評估一次；若要避免用 test set 調參，請只根據 validation 結果選設定，並將 test 結果保留作最終報告。

## 輸出內容

每次執行會建立 `result/<run_name>/`，包含：

| 檔案 | 內容 |
|---|---|
| `best.pt` | 依上述 checkpoint 規則選出的模型權重 |
| `history.csv` | 每 epoch 的 `epoch`、`train_accuracy`、`train_loss`、`validation_accuracy`、`validation_loss` |
| `training_accuracy.png` | Training accuracy curve |
| `training_loss.png` | Training loss curve |
| `validation_accuracy.png` | Validation accuracy curve |
| `validation_loss.png` | Validation loss curve |
| `highest_accuracy.txt` | Best validation loss、選中 checkpoint 的 validation accuracy/epoch、test accuracy 與執行參數 |
| `console.log` | 僅在使用 shell 重導向時產生；包含完整訓練輸出 |

## 常見問題

- **找不到 `.npz` 檔案**：確認目前工作目錄是專案根目錄，並確認 `lab2_EEG_classification/data/` 下四個檔案存在。
- **GPU 未啟用**：程式會自動退回 CPU；請確認已安裝與本機 CUDA 相容的 PyTorch 版本。
- **想保留每次完整終端輸出**：PowerShell 可使用 `python main.py ... *> result/<run_name>/console.log`，但請先建立該輸出資料夾，或直接從終端複製輸出。
