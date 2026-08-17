# HTNN for Sandbox

## Project configuration
- with N random fields
- under M events
- training for K times

## Project Structure

```text
project/
│
├── material/                         # 放訓練資料
│   ├── Random_field_1.dat
│   ├── ...
│   └── Random_field_N.dat
│
├── forward/                          # 模擬資料夾
│   ├── Event_1/                      # 注水事件
│   │   ├── SLE.exe
│   │   ├── SLE_io.dat
│   │   └── ...
│   │
│   ├── ...
│   │
│   └── Event_M/
│
├── experiment/                       # 實驗結果
│   ├── prior.csv                     # 先驗資訊
│   ├── model.ppt                     # 沙箱模型示意
│   └── Sandbox_grid.xlsx             # 沙箱網格定義
│
├── training_data/                    # 訓練資料
│   ├── X.hdf5
│   └── Y.hdf5
│
└── training_result/                  # 訓練結果
    └── train_folder/
        ├── checkpoint.pth
        └── metrics.csv
